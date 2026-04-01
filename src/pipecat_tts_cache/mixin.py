#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""TTS caching mixin for Fish Audio WebSocket TTS.

Design:
    Each run_tts("text") appends text to a FIFO queue (_pending_texts).
    Audio frames arrive in order via push_frame.  Inter-sentence silence
    frames (metadata["_tts_silence"]=True) delimit each sentence's audio.
    TTSStoppedFrame finalizes the last sentence (which has no trailing silence).

    Timeline:
        run_tts("A") → pending=["A"]
        run_tts("B") → pending=["A","B"]
        audio A ...
        silence frame  → STORE "A", pop, clear buffer
        audio B ...
        TTSStoppedFrame → STORE "B", pop, clear buffer

    On cache hit with no pending MISS entries, audio is replayed from cache
    (zero latency).  If MISS entries are pending, HIT is skipped to preserve
    playback order (Fish Audio returns audio in send order).
"""

from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from .backends.base import CacheBackend
from .key_generator import generate_cache_key
from .models import CachedAudioChunk, CachedTTSResponse, CachedWordTimestamp

_CACHE_ORIGIN_KEY = "_tts_cache_origin"
_BYTES_PER_PCM_SAMPLE = 2
_LOG_PREFIX = "[TTS_CACHE]"


class TTSCacheMixin:
    """Mixin that adds caching to any TTSService subclass.

    Usage: class CachedTTS(TTSCacheMixin, FishTTSService): pass
    """

    def __init__(
        self,
        *args,
        cache_backend: Optional[CacheBackend] = None,
        cache_ttl: Optional[int] = 5184000,
        cache_namespace: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._cache_backend = cache_backend
        self._cache_ttl = cache_ttl
        self._cache_namespace = cache_namespace

        self._cache_hits = 0
        self._cache_misses = 0
        self._pending_texts: List[str] = []
        self._current_audio_buffer: List[CachedAudioChunk] = []
        self._current_word_timestamps: List[Tuple[str, float]] = []
        self._serving_from_cache = False

        if cache_backend:
            logger.info(
                f"{_LOG_PREFIX} Enabled: backend={type(cache_backend).__name__}, "
                f"ttl={cache_ttl}s, namespace={cache_namespace or 'default'}"
            )

    def _generate_cache_key(self, text: str) -> str:
        voice_id = "default"
        model = "default"
        settings_dict: Dict[str, Any] = {}

        if hasattr(self, "_settings"):
            settings_obj = self._settings
            voice_id = getattr(settings_obj, "voice", None) or "default"
            model = getattr(settings_obj, "model", None) or "default"
            if hasattr(settings_obj, "given_fields"):
                settings_dict = settings_obj.given_fields()
        else:
            voice_id = getattr(self, "_voice_id", "default")
            model = getattr(self, "model_name", "default")
            settings_dict = getattr(self, "_settings", {})
            if not isinstance(settings_dict, dict):
                settings_dict = {}

        return generate_cache_key(
            text=text,
            voice_id=str(voice_id),
            model=str(model),
            sample_rate=getattr(self, "sample_rate", 16000),
            settings=settings_dict,
            namespace=self._cache_namespace,
        )

    # ------------------------------------------------------------------
    # run_tts: cache lookup / miss passthrough
    # ------------------------------------------------------------------

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        if not self._cache_backend:
            async for frame in super().run_tts(text, context_id):
                yield frame
            return

        cache_key = self._generate_cache_key(text)
        cached = await self._safe_cache_get(cache_key)

        if cached and not self._pending_texts:
            # HIT — replay from cache
            self._cache_hits += 1
            self._serving_from_cache = True
            total = self._cache_hits + self._cache_misses
            logger.info(
                f"{_LOG_PREFIX} HIT: '{text[:50]}' "
                f"({cached.metadata.get('audio_bytes', '?')}B, "
                f"{cached.total_duration_s:.1f}s) "
                f"[{self._cache_hits}/{total} = {self._cache_hits/total*100:.0f}%]"
            )
            try:
                async for frame in self._yield_cached_frames(cached, context_id):
                    yield frame
            finally:
                self._serving_from_cache = False
            return

        # MISS (or HIT skipped because earlier MISS is pending)
        if cached and self._pending_texts:
            logger.info(
                f"{_LOG_PREFIX} HIT_SKIP: '{text[:50]}' "
                f"({len(self._pending_texts)} MISS pending)"
            )

        self._cache_misses += 1
        total = self._cache_hits + self._cache_misses
        logger.info(
            f"{_LOG_PREFIX} MISS: '{text[:50]}' "
            f"[{self._cache_hits}/{total} = {self._cache_hits/total*100:.0f}%]"
        )

        self._pending_texts.append(text)

        try:
            async for frame in super().run_tts(text, context_id):
                yield frame
        except Exception:
            self._clear_state()
            raise

    # ------------------------------------------------------------------
    # push_frame: intercept audio for caching
    # ------------------------------------------------------------------

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        if not self._is_from_cache(frame) and self._pending_texts:
            if isinstance(frame, TTSAudioRawFrame):
                self._current_audio_buffer.append(
                    CachedAudioChunk(
                        audio=frame.audio,
                        sample_rate=frame.sample_rate,
                        num_channels=frame.num_channels,
                        pts=getattr(frame, "pts", None),
                    )
                )

                # Silence frame = end of current sentence's audio
                if frame.metadata.get("_tts_silence"):
                    await self._store_first_pending()

            elif isinstance(frame, TTSStoppedFrame):
                # Last sentence has no trailing silence — store on stop
                if self._pending_texts:
                    if self._current_audio_buffer:
                        await self._store_first_pending()
                    else:
                        # No audio arrived — discard pending text
                        discarded = self._pending_texts.pop(0)
                        logger.warning(
                            f"{_LOG_PREFIX} SKIP (no audio): '{discarded[:50]}'"
                        )

        await super().push_frame(frame, direction)

    # ------------------------------------------------------------------
    # Metrics: skip on cache hits
    # ------------------------------------------------------------------

    async def start_tts_usage_metrics(self, text: str):
        if self._serving_from_cache:
            return
        await super().start_tts_usage_metrics(text)

    # ------------------------------------------------------------------
    # Word timestamps
    # ------------------------------------------------------------------

    async def add_word_timestamps(
        self, word_times: List[Tuple[str, float]], context_id: Optional[str] = None
    ):
        filtered = [(w, t) for w, t in word_times if w not in ("TTSStoppedFrame", "Reset")]
        if self._pending_texts and filtered:
            self._current_word_timestamps.extend(filtered)
        if hasattr(super(), "add_word_timestamps"):
            await super().add_word_timestamps(word_times, context_id=context_id)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    async def _store_first_pending(self):
        """Store current audio buffer under the first pending text, then pop it."""
        if not self._pending_texts or not self._current_audio_buffer:
            return

        text = self._pending_texts.pop(0)
        all_audio = b"".join(c.audio for c in self._current_audio_buffer)
        sample_rate = self._current_audio_buffer[0].sample_rate
        num_channels = self._current_audio_buffer[0].num_channels
        duration = len(all_audio) / (sample_rate * _BYTES_PER_PCM_SAMPLE * num_channels)

        cache_key = self._generate_cache_key(text)

        try:
            response = CachedTTSResponse(
                audio_chunks=[CachedAudioChunk(all_audio, sample_rate, num_channels)],
                sample_rate=sample_rate,
                num_channels=num_channels,
                word_timestamps=[
                    CachedWordTimestamp(word=w, timestamp=t)
                    for w, t in self._current_word_timestamps
                ],
                total_duration_s=duration,
                metadata={"text": text, "audio_bytes": len(all_audio)},
            )
            success = await self._cache_backend.set(cache_key, response, ttl=self._cache_ttl)
            if success:
                logger.info(
                    f"{_LOG_PREFIX} STORE: '{text[:50]}' "
                    f"({len(all_audio)}B, {duration:.1f}s)"
                )
        except Exception as e:
            logger.error(f"{_LOG_PREFIX} STORE ERROR: '{text[:30]}': {e}")
        finally:
            self._current_audio_buffer.clear()
            self._current_word_timestamps.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _safe_cache_get(self, key: str) -> Optional[CachedTTSResponse]:
        try:
            return await self._cache_backend.get(key)
        except Exception as e:
            logger.warning(f"{_LOG_PREFIX} GET ERROR: {e}")
            return None

    async def _yield_cached_frames(
        self, cached: CachedTTSResponse, context_id: str
    ) -> AsyncGenerator[Frame, None]:
        for chunk in cached.audio_chunks:
            frame = TTSAudioRawFrame(
                audio=chunk.audio,
                sample_rate=chunk.sample_rate,
                num_channels=chunk.num_channels,
                context_id=context_id,
            )
            frame.metadata[_CACHE_ORIGIN_KEY] = True
            yield frame

    def _is_from_cache(self, frame: Frame) -> bool:
        return frame.metadata.get(_CACHE_ORIGIN_KEY, False)

    def _clear_state(self) -> None:
        self._pending_texts.clear()
        self._current_audio_buffer.clear()
        self._current_word_timestamps.clear()

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        if self._pending_texts:
            logger.info(
                f"{_LOG_PREFIX} INTERRUPT: clearing {len(self._pending_texts)} pending"
            )
            self._clear_state()
        if hasattr(super(), "_handle_interruption"):
            await super()._handle_interruption(frame, direction)

    # ------------------------------------------------------------------
    # Stats / management
    # ------------------------------------------------------------------

    async def get_cache_stats(self) -> Dict[str, Any]:
        total = self._cache_hits + self._cache_misses
        stats = {
            "enabled": self._cache_backend is not None,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total > 0 else 0.0,
            "total_requests": total,
        }
        if self._cache_backend:
            try:
                stats["backend"] = await self._cache_backend.get_stats()
            except Exception as e:
                logger.error(f"{_LOG_PREFIX} Stats error: {e}")
                stats["backend"] = {"error": str(e)}
        return stats

    async def clear_cache(self, namespace: Optional[str] = None) -> int:
        if not self._cache_backend:
            return 0
        try:
            cleared = await self._cache_backend.clear(namespace)
            logger.info(f"{_LOG_PREFIX} Cleared {cleared} entries")
            return cleared
        except Exception as e:
            logger.error(f"{_LOG_PREFIX} Clear error: {e}")
            return 0
