#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""TTS caching mixin for reducing API costs on repeated phrases.

Updated for compatibility with pipecat-ai >= 0.0.104 (context_id-aware API).

Key integration points with the current pipecat TTS pipeline:

    _push_tts_frames (base class) orchestrates TTS:
      1. create_audio_context(context_id) + append TTSStartedFrame  [if push_start_frame]
      2. start_ttfb_metrics / start_processing_metrics
      3. tts_process_generator(run_tts(...))  <- mixin overrides run_tts
      4. stop_processing_metrics
      5. TTSSentenceBoundaryFrame appended to audio context (per sentence)
      6. TTSStoppedFrame is pushed by stop-frame handler or audio-context teardown

    On cache hit the mixin must:
      - NOT yield TTSStartedFrame/TTSStoppedFrame (base already handles those)
      - Only yield TTSAudioRawFrame(s) -- they flow through tts_process_generator ->
        append_to_audio_context -> audio context queue -> push_frame
      - Skip TTS usage metrics (no real API call)
      - Flag the current request as a cache hit so metrics helpers can differentiate

Caching strategy:
    Per-sentence caching using TTSSentenceBoundaryFrame as the delimiter.
    Each sentence is cached under its own text key. Silence frames
    (metadata["_tts_silence"]) are excluded from cached audio.

    Fallback: if no TTSSentenceBoundaryFrame arrives before TTSStoppedFrame
    (e.g. older pipecat versions), all sentences are cached under a combined key.
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

try:
    from pipecat.frames.frames import TTSSentenceBoundaryFrame
except ImportError:
    TTSSentenceBoundaryFrame = None

from .backends.base import CacheBackend
from .key_generator import generate_cache_key
from .models import CachedAudioChunk, CachedTTSResponse, CachedWordTimestamp

_CACHE_ORIGIN_KEY = "_tts_cache_origin"
_BYTES_PER_PCM_SAMPLE = 2
_LOG_PREFIX = "[TTS_CACHE]"


class TTSCacheMixin:
    """Mixin that adds caching to any TTSService subclass.

    Usage: class CachedTTS(TTSCacheMixin, SomeTTSService): pass
    """

    def __init__(
        self,
        *args,
        cache_backend: Optional[CacheBackend] = None,
        cache_ttl: Optional[int] = 5184000,
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
        # Per-sentence caching state: texts queued by run_tts (FIFO),
        # audio buffer for the current sentence, word timestamps.
        self._pending_texts: List[str] = []
        self._current_audio_buffer: List[CachedAudioChunk] = []
        self._current_word_timestamps: List[Tuple[str, float]] = []
        # Track whether the current run_tts call is serving from cache so that
        # metric methods and push_frame can behave appropriately.
        self._serving_from_cache = False
        # Deferred finalize: when a TTSStoppedFrame arrives before audio
        # (e.g. WebSocket TTS with slow TTFB exceeding stop_frame_timeout),
        # keep _pending_texts alive for one cycle so late-arriving audio can
        # still be cached. Resolved at the start of the next run_tts call.
        self._deferred_finalize = False
        # Track when a sentence boundary has been received but the following
        # silence frame hasn't arrived yet.  Finalization is deferred until the
        # silence frame is captured so it is included in the cached entry.
        self._boundary_pending = False

        if self._enable_cache:
            logger.info(
                f"{_LOG_PREFIX} Enabled: backend={type(cache_backend).__name__}, "
                f"ttl={cache_ttl}s, namespace={cache_namespace or 'default'}"
            )
        else:
            logger.debug(f"{_LOG_PREFIX} Disabled: no backend provided")

    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for the current TTS request."""
        # Modern pipecat stores voice/model in _settings (a ServiceSettings dataclass).
        # Fall back to legacy _voice_id / model_name for older service implementations.
        voice_id = "default"
        model = "default"
        settings_dict: Dict[str, Any] = {}

        if hasattr(self, "_settings"):
            settings_obj = self._settings
            voice_id = getattr(settings_obj, "voice", None) or "default"
            model = getattr(settings_obj, "model", None) or "default"
            # given_fields() returns a dict of all non-NOT_GIVEN fields
            if hasattr(settings_obj, "given_fields"):
                settings_dict = settings_obj.given_fields()
            else:
                settings_dict = {}
        else:
            # Legacy fallback
            voice_id = getattr(self, "_voice_id", "default")
            model = getattr(self, "model_name", "default")
            settings_dict = getattr(self, "_settings", {})
            if not isinstance(settings_dict, dict):
                settings_dict = {}

        sample_rate = getattr(self, "sample_rate", 16000)

        return generate_cache_key(
            text=text,
            voice_id=str(voice_id),
            model=str(model),
            sample_rate=sample_rate,
            settings=settings_dict,
            namespace=self._cache_namespace,
        )

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Run TTS with caching support.

        On cache miss: delegates to the real TTS service and records audio for caching.
        On cache hit: yields only TTSAudioRawFrame(s). The base class handles
        TTSStartedFrame/TTSStoppedFrame and metrics around the run_tts call.

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
            if self._current_audio_buffer and self._pending_texts:
                deferred_text = self._pending_texts[0][:50]
                audio_bytes = sum(len(c.audio) for c in self._current_audio_buffer)
                await self._finalize_remaining()
                logger.info(
                    f"{_LOG_PREFIX} DEFERRED STORE: '{deferred_text}...' "
                    f"({audio_bytes} bytes arrived late)"
                )
            elif self._pending_texts:
                logger.warning(
                    f"{_LOG_PREFIX} DEFERRED DISCARD: '{self._pending_texts[0][:50]}...' "
                    f"(audio never arrived)"
                )
                self._clear_batch_state()
            self._deferred_finalize = False

        # Check cache for this individual text first
        cache_key = self._generate_cache_key(text)
        # Debug: log key generation inputs to diagnose cache key mismatch
        _s = getattr(self, "_settings", None)
        _dbg_voice = getattr(_s, "voice", "?") if _s else "?"
        _dbg_model = getattr(_s, "model", "?") if _s else "?"
        _dbg_given = _s.given_fields() if _s and hasattr(_s, "given_fields") else {}
        _dbg_ns = getattr(self, "_cache_namespace", None)
        logger.debug(
            f"{_LOG_PREFIX} cache_key={cache_key[:16]}... text='{text[:40]}' "
            f"voice={_dbg_voice} model={_dbg_model} sr={getattr(self, 'sample_rate', '?')} "
            f"given={_dbg_given} ns={_dbg_ns}"
        )
        cached_response = await self._safe_cache_get(cache_key)

        if cached_response:
            # If there are pending MISS sentences whose audio hasn't arrived yet,
            # skip the cache and send to Fish Audio instead. This preserves playback
            # order: Fish Audio returns audio in send order, so all sentences in the
            # same turn play back sequentially. Using cached audio here would cause
            # the HIT sentence to play before the earlier MISS sentence.
            if self._pending_texts:
                logger.info(
                    f"{_LOG_PREFIX} HIT (skip, {len(self._pending_texts)} MISS pending): "
                    f"'{text[:50]}...' — sending to TTS to preserve order"
                )
            else:
                self._cache_hits += 1
                self._serving_from_cache = True
                total = self._cache_hits + self._cache_misses
                hit_rate = self._cache_hits / total * 100
                audio_bytes = sum(len(c.audio) for c in cached_response.audio_chunks)
                logger.info(
                    f"{_LOG_PREFIX} HIT: '{text[:50]}...' "
                    f"({audio_bytes} bytes, {cached_response.total_duration_s:.1f}s) "
                    f"[{self._cache_hits}/{total} = {hit_rate:.0f}%]"
                )
                try:
                    async for frame in self._yield_cached_frames(cached_response, context_id):
                        yield frame
                finally:
                    self._serving_from_cache = False
                return

        self._cache_misses += 1
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total * 100
        logger.info(
            f"{_LOG_PREFIX} MISS: '{text[:50]}...' "
            f"[{self._cache_hits}/{total} = {hit_rate:.0f}%]"
        )

        self._pending_texts.append(text)

        try:
            async for frame in super().run_tts(text, context_id):
                yield frame
        except Exception as e:
            logger.error(f"{_LOG_PREFIX} TTS generation failed: {e}")
            self._clear_batch_state()
            raise

    async def _safe_cache_get(self, key: str) -> Optional[CachedTTSResponse]:
        """Get from cache with error handling."""
        try:
            return await self._cache_backend.get(key)
        except Exception as e:
            logger.warning(f"{_LOG_PREFIX} GET ERROR: {e}")
            return None

    async def _yield_cached_frames(
        self, cached: CachedTTSResponse, context_id: str
    ) -> AsyncGenerator[Frame, None]:
        """Replay audio frames from cache.

        Only yields TTSAudioRawFrame(s). The base class already handles
        TTSStartedFrame (via push_start_frame) and TTSStoppedFrame (via the
        stop-frame handler / audio-context teardown), so we must NOT yield
        those here to avoid duplicates.
        """
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
        """Check if a frame originated from cache replay."""
        return frame.metadata.get(_CACHE_ORIGIN_KEY, False)

    # ------------------------------------------------------------------
    # Metrics: skip on cache hits to avoid misleading numbers
    # ------------------------------------------------------------------

    async def start_tts_usage_metrics(self, text: str):
        """Skip TTS usage metrics on cache hits (no real API call)."""
        if self._serving_from_cache:
            return
        await super().start_tts_usage_metrics(text)

    # ------------------------------------------------------------------
    # Frame interception for cache population
    # ------------------------------------------------------------------

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Override push_frame to intercept audio frames for caching."""
        should_finalize = False

        if not self._is_from_cache(frame) and self._pending_texts:
            # TTSSentenceBoundaryFrame: mark boundary pending so the next
            # silence frame is included in the current sentence's cache entry
            # before we finalize.
            if TTSSentenceBoundaryFrame is not None and isinstance(
                frame, TTSSentenceBoundaryFrame
            ):
                self._boundary_pending = True
                # Don't pass boundary frames downstream
                return

            if isinstance(frame, TTSAudioRawFrame):
                chunk = CachedAudioChunk(
                    audio=frame.audio,
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                    pts=getattr(frame, "pts", None),
                )
                self._current_audio_buffer.append(chunk)

                # If a boundary was pending and this is the silence frame that
                # follows it, finalize the sentence now that silence is captured.
                if self._boundary_pending and frame.metadata.get("_tts_silence"):
                    self._boundary_pending = False
                    await self._cache_current_sentence()
                # Non-silence audio after a boundary (edge case: no silence
                # frame was emitted) — finalize the previous sentence anyway.
                elif self._boundary_pending:
                    self._boundary_pending = False
                    await self._cache_current_sentence()

            elif isinstance(frame, TTSStoppedFrame):
                # If a boundary was pending when stop arrives, finalize first.
                if self._boundary_pending:
                    self._boundary_pending = False
                    await self._cache_current_sentence()
                should_finalize = True

        # Let the base class handle the frame first.  For TTSStoppedFrame the
        # base TTSService.push_frame may re-entrantly push a silence
        # TTSAudioRawFrame (push_silence_after_stop).  By deferring
        # finalization the silence frame is captured before we write to cache.
        await super().push_frame(frame, direction)

        if should_finalize:
            await self._finalize_remaining()

    async def add_word_timestamps(
        self, word_times: List[Tuple[str, float]], context_id: Optional[str] = None
    ):
        """Intercept word timestamps for caching."""
        filtered_times = [(w, t) for w, t in word_times if w not in ("TTSStoppedFrame", "Reset")]

        if self._pending_texts and filtered_times:
            self._current_word_timestamps.extend(filtered_times)

        if hasattr(super(), "add_word_timestamps"):
            await super().add_word_timestamps(word_times, context_id=context_id)

    # ------------------------------------------------------------------
    # Cache storage
    # ------------------------------------------------------------------

    async def _cache_current_sentence(self):
        """Cache the audio buffer for the current (first pending) sentence."""
        if not self._pending_texts:
            return

        text = self._pending_texts.pop(0)

        if not self._current_audio_buffer:
            logger.warning(f"{_LOG_PREFIX} SKIP (no audio): '{text[:50]}...'")
            self._current_word_timestamps.clear()
            return

        await self._store_cache_entry(text)

    async def _finalize_remaining(self):
        """Cache any remaining pending texts on TTSStoppedFrame.

        If TTSSentenceBoundaryFrame already handled all sentences, there is
        nothing left to do.  Otherwise (older pipecat without boundary frames),
        fall back to combined-key caching.
        """
        if not self._pending_texts:
            self._current_audio_buffer.clear()
            self._current_word_timestamps.clear()
            return

        if not self._current_audio_buffer:
            if not self._deferred_finalize:
                # First time: defer — audio may arrive late (WebSocket TTS).
                self._deferred_finalize = True
                logger.info(
                    f"{_LOG_PREFIX} DEFER: {len(self._pending_texts)} text(s) "
                    f"(no audio yet, will retry next run_tts)"
                )
                return
            else:
                # Second time: audio never arrived — discard.
                self._deferred_finalize = False
                logger.warning(
                    f"{_LOG_PREFIX} DISCARD: {len(self._pending_texts)} text(s) "
                    f"(audio never arrived after defer)"
                )
                self._clear_batch_state()
                return

        if len(self._pending_texts) == 1:
            cache_text = self._pending_texts[0]
        else:
            cache_text = " ".join(self._pending_texts)
            logger.info(
                f"{_LOG_PREFIX} COMBINE: {len(self._pending_texts)} sentences "
                f"(no boundary frames): '{cache_text[:80]}...'"
            )

        await self._store_cache_entry(cache_text)
        self._pending_texts.clear()

    async def _store_cache_entry(self, cache_text: str):
        """Store collected audio buffer under the given text key."""
        all_audio = b"".join(chunk.audio for chunk in self._current_audio_buffer)
        sample_rate = self._current_audio_buffer[0].sample_rate
        num_channels = self._current_audio_buffer[0].num_channels
        duration = self._audio_duration(all_audio, sample_rate, num_channels)

        timestamps = [
            CachedWordTimestamp(word=w, timestamp=t) for w, t in self._current_word_timestamps
        ]

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
                },
            )

            success = await self._cache_backend.set(
                cache_key, cached_response, ttl=self._cache_ttl
            )
            if success:
                logger.info(
                    f"{_LOG_PREFIX} STORE: '{cache_text[:50]}...' "
                    f"({len(all_audio)} bytes, {duration:.1f}s)"
                )
        except Exception as e:
            logger.error(f"{_LOG_PREFIX} STORE ERROR: '{cache_text[:30]}...': {e}")

        self._current_audio_buffer.clear()
        self._current_word_timestamps.clear()

    @staticmethod
    def _audio_duration(audio: bytes, sample_rate: int, num_channels: int) -> float:
        return len(audio) / (sample_rate * _BYTES_PER_PCM_SAMPLE * num_channels)

    def _clear_batch_state(self) -> None:
        """Clear all batch-related state."""
        self._pending_texts.clear()
        self._current_audio_buffer.clear()
        self._current_word_timestamps.clear()
        self._boundary_pending = False

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        """Handle interruptions during TTS generation."""
        self._deferred_finalize = False
        self._boundary_pending = False
        if self._pending_texts:
            logger.info(
                f"{_LOG_PREFIX} INTERRUPT: clearing {len(self._pending_texts)} pending text(s)"
            )
            self._clear_batch_state()

        if hasattr(super(), "_handle_interruption"):
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
                logger.error(f"{_LOG_PREFIX} Stats error: {e}")
                stats["backend"] = {"error": str(e)}

        return stats

    async def clear_cache(self, namespace: Optional[str] = None) -> int:
        """Clear cache entries."""
        if not self._cache_backend:
            logger.warning(f"{_LOG_PREFIX} Cannot clear: no backend configured")
            return 0

        try:
            cleared = await self._cache_backend.clear(namespace)
            logger.info(f"{_LOG_PREFIX} Cleared {cleared} entries")
            return cleared
        except Exception as e:
            logger.error(f"{_LOG_PREFIX} Clear error: {e}")
            return 0
