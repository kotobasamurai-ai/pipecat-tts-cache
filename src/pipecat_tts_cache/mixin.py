#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""TTS caching mixin for reducing API costs on repeated phrases.

Updated for compatibility with pipecat-ai >= 0.0.104 (context_id-aware API).

Key integration points with the current pipecat TTS pipeline:

    _synthesize_text (base class) orchestrates TTS:
      1. create_audio_context(context_id) + append TTSStartedFrame  [if push_start_frame]
      2. start_ttfb_metrics / start_processing_metrics
      3. tts_process_generator(run_tts(...))  <- mixin overrides run_tts
      4. stop_processing_metrics
      5. TTSStoppedFrame is pushed by stop-frame handler or audio-context teardown

    On cache hit the mixin must:
      - NOT yield TTSStartedFrame/TTSStoppedFrame (base already handles those)
      - Only yield TTSAudioRawFrame(s) -- they flow through tts_process_generator ->
        append_to_audio_context -> audio context queue -> push_frame
      - Skip TTS usage metrics (no real API call)
      - Flag the current request as a cache hit so metrics helpers can differentiate

Caching strategy:
    - Single sentence: cached under that sentence's text key.
    - Multiple sentences in one turn (batched by the pipeline before
      TTSStoppedFrame): cached under the combined text key. This handles
      WebSocket TTS services where audio for all sentences arrives as one
      continuous stream that cannot be split per sentence.
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


class TTSCacheMixin:
    """Mixin that adds caching to any TTSService subclass.

    Usage: class CachedTTS(TTSCacheMixin, SomeTTSService): pass
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
        # Track whether the current run_tts call is serving from cache so that
        # metric methods and push_frame can behave appropriately.
        self._serving_from_cache = False

        if self._enable_cache:
            logger.info(
                f"TTS caching enabled: backend={type(cache_backend).__name__}, "
                f"ttl={cache_ttl}s, namespace={cache_namespace or 'default'}, "
            )
        else:
            logger.debug("TTS caching disabled: no backend provided")

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

        # Check cache for this individual text first
        cache_key = self._generate_cache_key(text)
        cached_response = await self._safe_cache_get(cache_key)

        if cached_response:
            self._cache_hits += 1
            self._serving_from_cache = True
            logger.debug(
                f"Cache hit: '{text[:50]}...' ({len(cached_response.audio_chunks)} chunks)"
            )
            try:
                async for frame in self._yield_cached_frames(cached_response, context_id):
                    yield frame
            finally:
                self._serving_from_cache = False
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

        if hasattr(super(), "add_word_timestamps"):
            await super().add_word_timestamps(word_times, context_id=context_id)

    # ------------------------------------------------------------------
    # Cache storage
    # ------------------------------------------------------------------

    async def _finalize_batch(self):
        """Cache the collected audio for the current batch of texts."""
        if not self._batch_texts:
            return

        if not self._batch_audio_buffer:
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
        if self._batch_texts:
            logger.debug(
                f"Interruption - clearing {len(self._batch_texts)} pending cache text(s)"
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
