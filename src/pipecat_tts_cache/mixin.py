#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""TTS caching mixin for reducing API costs on repeated phrases.

Updated for compatibility with pipecat-ai >= 0.0.104 (context_id-aware API).
"""

import inspect
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


class TTSCacheMixin:
    """Mixin that adds caching to any TTSService subclass.

    Usage: class CachedTTS(TTSCacheMixin, SomeTTSService): pass

    Caching strategy:
    - Single sentence: cached under that sentence's text key.
    - Multiple sentences in one turn (batched by the pipeline before
      TTSStoppedFrame): cached under the combined text key. This handles
      WebSocket TTS services where audio for all sentences arrives as one
      continuous stream that cannot be split per sentence.
    """

    def __init__(
        self,
        *args,
        cache_backend: Optional[CacheBackend] = None,
        cache_ttl: Optional[int] = 86400,
        cache_namespace: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize TTS cache mixin.

        Args:
            *args: Positional arguments passed to parent class.
            cache_backend: Cache backend instance. If None, caching is disabled.
            cache_ttl: Time-to-live for cache entries in seconds.
            cache_namespace: Optional namespace prefix for cache keys.
            **kwargs: Keyword arguments passed to parent class.
        """
        super().__init__(*args, **kwargs)
        self._cache_backend = cache_backend
        self._cache_ttl = cache_ttl
        self._cache_namespace = cache_namespace
        self._enable_cache = cache_backend is not None

        self._cache_hits = 0
        self._cache_misses = 0
        self._batch_texts: List[str] = []
        self._batch_audio_buffer: List[CachedAudioChunk] = []
        self._batch_word_timestamps: List[Tuple[str, float]] = []

        # Cache capability checks resolved once at init (MRO is static).
        self._has_parent_add_word_timestamps = hasattr(super(), "add_word_timestamps")
        self._has_parent_handle_interruption = hasattr(super(), "_handle_interruption")
        if hasattr(self, "_settings") and hasattr(self._settings, "given_fields"):
            self._settings_mode = "modern"
        elif hasattr(self, "_settings"):
            self._settings_mode = "modern_no_given_fields"
        else:
            self._settings_mode = "legacy"

        # Deferred finalize: when a TTSStoppedFrame arrives before audio
        # (e.g. WebSocket TTS with slow TTFB exceeding stop_frame_timeout),
        # keep _batch_texts alive for one cycle so late-arriving audio can
        # still be cached. Resolved at the start of the next run_tts call.
        self._deferred_finalize = False

        if self._enable_cache:
            logger.info(
                f"TTS caching enabled: backend={type(cache_backend).__name__}, "
                f"ttl={cache_ttl}s, namespace={cache_namespace or 'default'}, "
            )
        else:
            logger.debug("TTS caching disabled: no backend provided")

    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for the current TTS request."""
        if self._settings_mode == "modern":
            settings_obj = self._settings
            voice_id = getattr(settings_obj, "voice", None) or "default"
            model = getattr(settings_obj, "model", None) or "default"
            settings_dict = settings_obj.given_fields()
        elif self._settings_mode == "modern_no_given_fields":
            settings_obj = self._settings
            voice_id = getattr(settings_obj, "voice", None) or "default"
            model = getattr(settings_obj, "model", None) or "default"
            settings_dict = {}
        else:
            voice_id = getattr(self, "_voice_id", "default")
            model = getattr(self, "model_name", "default")
            settings_dict = {}

        return generate_cache_key(
            text=text,
            voice_id=str(voice_id),
            model=str(model),
            sample_rate=getattr(self, "sample_rate", 16000),
            settings=settings_dict,
            namespace=self._cache_namespace,
        )

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Run TTS with caching support.

        Args:
            text: The text to synthesize.
            context_id: Unique identifier for this TTS context.
        """
        if not self._enable_cache:
            async for frame in super().run_tts(text, context_id):
                yield frame
            return

        # Resolve any deferred cache from a previous run_tts whose audio
        # arrived after TTSStoppedFrame (WebSocket TTS with slow TTFB).
        if self._deferred_finalize:
            if self._batch_audio_buffer and self._batch_texts:
                deferred_text = self._batch_texts[0][:50]
                await self._finalize_batch()
                logger.debug(f"Deferred cache saved: '{deferred_text}...'")
            elif self._batch_texts:
                logger.debug(
                    f"Deferred cache discarded (no audio): '{self._batch_texts[0][:50]}...'"
                )
                self._clear_batch_state()
            self._deferred_finalize = False

        # Check cache for this individual text first
        cache_key = self._generate_cache_key(text)
        cached_response = await self._safe_cache_get(cache_key)

        if cached_response:
            self._cache_hits += 1
            logger.debug(
                f"Cache hit: '{text[:50]}...' ({len(cached_response.audio_chunks)} chunks)"
            )
            async for frame in self._yield_cached_frames(cached_response, context_id):
                yield frame
            return

        self._cache_misses += 1
        logger.debug(f"Cache miss: '{text[:50]}...'")

        self._batch_texts.append(text)

        try:
            async for frame in super().run_tts(text, context_id):
                yield frame
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            self._clear_batch_state()
            raise

    async def _safe_cache_get(self, key: str) -> Optional[CachedTTSResponse]:
        """Get from cache with error handling."""
        try:
            return await self._cache_backend.get(key)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None

    async def _yield_cached_frames(
        self, cached: CachedTTSResponse, context_id: str
    ) -> AsyncGenerator[Frame, None]:
        """Replay frames from cache with proper context_id."""
        started_frame = TTSStartedFrame(context_id=context_id)
        started_frame.metadata[_CACHE_ORIGIN_KEY] = True
        yield started_frame

        if hasattr(self, "start_word_timestamps") and cached.word_timestamps:
            result = self.start_word_timestamps()
            if inspect.iscoroutine(result):
                await result
            word_times: List[Tuple[str, float]] = [
                (wt.word, wt.timestamp) for wt in cached.word_timestamps
            ]
            if hasattr(self, "add_word_timestamps"):
                await self._add_word_timestamps_from_cache(word_times, context_id)

        for chunk in cached.audio_chunks:
            frame = TTSAudioRawFrame(
                audio=chunk.audio,
                sample_rate=chunk.sample_rate,
                num_channels=chunk.num_channels,
                context_id=context_id,
            )
            frame.metadata[_CACHE_ORIGIN_KEY] = True
            yield frame

        stopped_frame = TTSStoppedFrame(context_id=context_id)
        stopped_frame.metadata[_CACHE_ORIGIN_KEY] = True
        yield stopped_frame

    async def _add_word_timestamps_from_cache(
        self, word_times: List[Tuple[str, float]], context_id: str
    ):
        """Add word timestamps from cache without collecting them."""
        if self._has_parent_add_word_timestamps:
            await super().add_word_timestamps(word_times, context_id=context_id)

    def _is_from_cache(self, frame: Frame) -> bool:
        """Check if a frame originated from cache replay."""
        return frame.metadata.get(_CACHE_ORIGIN_KEY, False)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Override push_frame to intercept audio frames for caching."""
        if not self._is_from_cache(frame) and self._batch_texts:
            if isinstance(frame, TTSAudioRawFrame):
                chunk = CachedAudioChunk(
                    audio=frame.audio,
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                    pts=getattr(frame, "pts", None),
                )
                self._batch_audio_buffer.append(chunk)

            elif isinstance(frame, TTSStoppedFrame):
                await self._finalize_batch()

        await super().push_frame(frame, direction)

    async def add_word_timestamps(
        self, word_times: List[Tuple[str, float]], context_id: Optional[str] = None
    ):
        """Intercept word timestamps for caching."""
        filtered_times = [(w, t) for w, t in word_times if w not in ("TTSStoppedFrame", "Reset")]

        if self._batch_texts and filtered_times:
            self._batch_word_timestamps.extend(filtered_times)

        if self._has_parent_add_word_timestamps:
            await super().add_word_timestamps(word_times, context_id=context_id)

    async def _finalize_batch(self):
        """Cache the collected audio for the current batch of texts."""
        if not self._batch_texts:
            return

        if not self._batch_audio_buffer:
            if not self._deferred_finalize:
                # First time: defer — audio may arrive late (WebSocket TTS).
                self._deferred_finalize = True
                logger.debug(
                    f"No audio yet for {len(self._batch_texts)} text(s), "
                    f"deferring cache to next run_tts"
                )
                return
            else:
                # Second time: audio never arrived — discard.
                self._deferred_finalize = False
                logger.warning(
                    f"No audio collected for {len(self._batch_texts)} text(s), skipping cache"
                )
                self._clear_batch_state()
                return

        all_audio = b"".join(chunk.audio for chunk in self._batch_audio_buffer)
        sample_rate = self._batch_audio_buffer[0].sample_rate
        num_channels = self._batch_audio_buffer[0].num_channels
        duration = self._audio_duration(all_audio, sample_rate, num_channels)

        timestamps = [
            CachedWordTimestamp(word=w, timestamp=t) for w, t in self._batch_word_timestamps
        ]

        # Determine cache key: single text uses its own key, multiple texts
        # use a combined key so the same multi-sentence turn is a cache hit.
        if len(self._batch_texts) == 1:
            cache_text = self._batch_texts[0]
        else:
            cache_text = " ".join(self._batch_texts)
            logger.debug(
                f"Caching {len(self._batch_texts)} sentences as combined entry: "
                f"'{cache_text[:80]}...'"
            )

        cache_key = self._generate_cache_key(cache_text)

        try:
            cached_response = CachedTTSResponse(
                audio_chunks=[CachedAudioChunk(all_audio, sample_rate, num_channels)],
                sample_rate=sample_rate,
                num_channels=num_channels,
                word_timestamps=timestamps,
                total_duration_s=duration,
                metadata={
                    "text": cache_text,
                    "audio_bytes": len(all_audio),
                    "word_count": len(timestamps),
                    "sentence_count": len(self._batch_texts),
                },
            )

            success = await self._cache_backend.set(
                cache_key, cached_response, ttl=self._cache_ttl
            )
            if success:
                logger.debug(f"Cached: '{cache_text[:50]}...' ({len(all_audio)} bytes)")
        except Exception as e:
            logger.error(f"Error caching '{cache_text[:30]}...': {e}")

        self._clear_batch_state()

    @staticmethod
    def _audio_duration(audio: bytes, sample_rate: int, num_channels: int) -> float:
        return len(audio) / (sample_rate * _BYTES_PER_PCM_SAMPLE * num_channels)

    def _clear_batch_state(self) -> None:
        """Clear all batch-related state."""
        self._batch_texts.clear()
        self._batch_audio_buffer.clear()
        self._batch_word_timestamps.clear()

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        """Handle interruptions during TTS generation."""
        self._deferred_finalize = False
        if self._batch_texts:
            logger.debug(
                f"Interruption - clearing {len(self._batch_texts)} pending cache text(s)"
            )
            self._clear_batch_state()

        if self._has_parent_handle_interruption:
            await super()._handle_interruption(frame, direction)

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0

        stats = {
            "enabled": self._enable_cache,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total,
        }

        if self._cache_backend:
            try:
                backend_stats = await self._cache_backend.get_stats()
                stats["backend"] = backend_stats
            except Exception as e:
                logger.error(f"Error getting backend stats: {e}")
                stats["backend"] = {"error": str(e)}

        return stats

    async def clear_cache(self, namespace: Optional[str] = None) -> int:
        """Clear cache entries."""
        if not self._cache_backend:
            logger.warning("Cannot clear cache: no backend configured")
            return 0

        try:
            cleared = await self._cache_backend.clear(namespace)
            logger.info(f"Cleared {cleared} cache entries")
            return cleared
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0
