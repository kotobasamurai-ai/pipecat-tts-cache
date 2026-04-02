#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for TTS caching mixin compatibility with current pipecat API.

Tests cover:
- Cache miss → stores audio on TTSStoppedFrame, cache hit → replays
- run_tts passes context_id through to parent
- Replayed frames carry context_id
- Interruption clears batch state
- Word timestamps forwarded with context_id
- Silence frames buffered as audio (not used as delimiters)
- Cache key generation uses _settings.voice / _settings.model
- Cache disabled when no backend provided
- Cache stats tracking
- Push_frame intercepts audio and stop frames for caching
- _serving_from_cache flag and metrics skipping
- No duplicate Started/Stopped frames on cache hit
"""

import unittest
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import pytest

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService

from pipecat_tts_cache import TTSCacheMixin
from pipecat_tts_cache.backends.memory import MemoryCacheBackend
from pipecat_tts_cache.key_generator import generate_cache_key
from pipecat_tts_cache.models import CachedAudioChunk, CachedTTSResponse, CachedWordTimestamp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FAKE_AUDIO = b"\x00\x01" * 320
_FAKE_AUDIO_2 = b"\x01\x00" * 320
_SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Mock TTS service (HTTP-style, yields frames synchronously from run_tts)
# ---------------------------------------------------------------------------


class MockHttpTTSService(TTSService):
    """Minimal HTTP-style TTS service for testing."""

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )
        self.run_tts_calls: List[Tuple[str, str]] = []

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        self.run_tts_calls.append((text, context_id))
        yield TTSAudioRawFrame(
            audio=_FAKE_AUDIO,
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            context_id=context_id,
        )


class CachedMockTTS(TTSCacheMixin, MockHttpTTSService):
    """Mock TTS with caching enabled."""

    pass


# ---------------------------------------------------------------------------
# Tests: Cache key generation
# ---------------------------------------------------------------------------


class TestCacheKeyGeneration(unittest.TestCase):
    """Tests for deterministic cache key generation."""

    def test_same_input_same_key(self):
        key1 = generate_cache_key("hello", "voice1")
        key2 = generate_cache_key("hello", "voice1")
        assert key1 == key2

    def test_different_text_different_key(self):
        key1 = generate_cache_key("hello", "voice1")
        key2 = generate_cache_key("world", "voice1")
        assert key1 != key2

    def test_different_voice_different_key(self):
        key1 = generate_cache_key("hello", "voice1")
        key2 = generate_cache_key("hello", "voice2")
        assert key1 != key2

    def test_whitespace_normalization(self):
        key1 = generate_cache_key("hello  world", "v")
        key2 = generate_cache_key("hello world", "v")
        assert key1 == key2

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="empty text"):
            generate_cache_key("", "v")

    def test_key_is_sha256_hex(self):
        key = generate_cache_key("hello", "v")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


# ---------------------------------------------------------------------------
# Tests: Memory cache backend
# ---------------------------------------------------------------------------


class TestMemoryCacheBackend:
    @pytest.fixture
    def backend(self):
        return MemoryCacheBackend(max_size=5)

    @pytest.fixture
    def sample_response(self):
        return CachedTTSResponse(
            audio_chunks=[CachedAudioChunk(audio=_FAKE_AUDIO, sample_rate=16000, num_channels=1)],
            sample_rate=16000,
            num_channels=1,
            total_duration_s=0.04,
        )

    @pytest.mark.asyncio
    async def test_set_and_get(self, backend, sample_response):
        await backend.set("key1", sample_response, ttl=3600)
        result = await backend.get("key1")
        assert result is not None
        assert result.sample_rate == 16000
        assert len(result.audio_chunks) == 1

    @pytest.mark.asyncio
    async def test_get_missing_key(self, backend):
        result = await backend.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, backend, sample_response):
        await backend.set("key1", sample_response)
        assert await backend.delete("key1")
        assert await backend.get("key1") is None

    @pytest.mark.asyncio
    async def test_exists(self, backend, sample_response):
        assert not await backend.exists("key1")
        await backend.set("key1", sample_response)
        assert await backend.exists("key1")

    @pytest.mark.asyncio
    async def test_lru_eviction(self, backend, sample_response):
        for i in range(6):
            await backend.set(f"key{i}", sample_response)
        # key0 should have been evicted (max_size=5)
        assert await backend.get("key0") is None
        assert await backend.get("key5") is not None

    @pytest.mark.asyncio
    async def test_clear_all(self, backend, sample_response):
        await backend.set("a", sample_response)
        await backend.set("b", sample_response)
        cleared = await backend.clear()
        assert cleared == 2
        assert await backend.get("a") is None

    @pytest.mark.asyncio
    async def test_stats(self, backend, sample_response):
        await backend.set("key1", sample_response)
        await backend.get("key1")  # hit
        await backend.get("missing")  # miss
        stats = await backend.get_stats()
        assert stats["type"] == "memory"
        assert stats["size"] == 1
        assert stats["backend_hits"] == 1
        assert stats["backend_misses"] == 1

    @pytest.mark.asyncio
    async def test_ttl_zero_means_no_expiry(self, backend, sample_response):
        # ttl=0 → expiry=0.0 which is falsy, so no TTL check
        await backend.set("key1", sample_response, ttl=0)
        result = await backend.get("key1")
        assert result is not None


# ---------------------------------------------------------------------------
# Tests: Mixin integration
# ---------------------------------------------------------------------------


class TestTTSCacheMixinIntegration:
    """Test the mixin wired into a mock TTS service."""

    @pytest.fixture
    def backend(self):
        return MemoryCacheBackend(max_size=100)

    def _make_service(self, backend):
        return CachedMockTTS(cache_backend=backend)

    @pytest.mark.asyncio
    async def test_cache_disabled_without_backend(self):
        """When no backend is provided, caching is disabled."""
        svc = CachedMockTTS(cache_backend=None)
        assert svc._cache_backend is None

    @pytest.mark.asyncio
    async def test_cache_enabled_with_backend(self, backend):
        svc = self._make_service(backend)
        assert svc._cache_backend is not None

    @pytest.mark.asyncio
    async def test_run_tts_passes_context_id_to_parent(self, backend):
        """run_tts must forward context_id to the parent TTSService."""
        svc = self._make_service(backend)
        context_id = "test-ctx-123"
        frames = []
        async for frame in svc.run_tts("hello world", context_id):
            frames.append(frame)

        assert len(svc.run_tts_calls) == 1
        assert svc.run_tts_calls[0] == ("hello world", context_id)

    @pytest.mark.asyncio
    async def test_cache_miss_stores_then_hit_replays(self, backend):
        """First call is a miss (calls parent), second is a hit (replays from cache)."""
        svc = self._make_service(backend)
        ctx = "ctx-1"

        # First call — cache miss
        miss_frames = []
        async for f in svc.run_tts("hello", ctx):
            miss_frames.append(f)
        assert len(svc.run_tts_calls) == 1

        # Simulate the pipeline pushing audio and stop frames through push_frame
        for f in miss_frames:
            if isinstance(f, TTSAudioRawFrame):
                await svc.push_frame(f, FrameDirection.DOWNSTREAM)
        await svc.push_frame(TTSStoppedFrame(context_id=ctx), FrameDirection.DOWNSTREAM)

        # Verify cache populated
        stats = await backend.get_stats()
        assert stats["size"] == 1

        # Second call — should be a cache hit
        hit_frames = []
        async for f in svc.run_tts("hello", ctx):
            hit_frames.append(f)

        # Parent should NOT have been called a second time
        assert len(svc.run_tts_calls) == 1
        assert svc._cache_hits == 1
        assert svc._cache_misses == 1

    @pytest.mark.asyncio
    async def test_cached_frames_have_context_id(self, backend):
        """Replayed audio frames from cache must carry the correct context_id."""
        svc = self._make_service(backend)

        # Populate cache
        frames = []
        async for f in svc.run_tts("test phrase", "orig-ctx"):
            frames.append(f)
        for f in frames:
            if isinstance(f, TTSAudioRawFrame):
                await svc.push_frame(f, FrameDirection.DOWNSTREAM)
        await svc.push_frame(TTSStoppedFrame(context_id="orig-ctx"), FrameDirection.DOWNSTREAM)

        # Replay with a DIFFERENT context_id
        replay_ctx = "replay-ctx-456"
        replayed = []
        async for f in svc.run_tts("test phrase", replay_ctx):
            replayed.append(f)

        # All replayed audio frames should have the new context_id
        for f in replayed:
            if isinstance(f, TTSAudioRawFrame):
                assert f.context_id == replay_ctx, (
                    f"{type(f).__name__}.context_id={f.context_id!r}, expected {replay_ctx!r}"
                )

    @pytest.mark.asyncio
    async def test_interruption_clears_batch_state(self, backend):
        """Interruption should clear pending batch state."""
        svc = self._make_service(backend)

        # Start a run_tts to create batch state
        async for _ in svc.run_tts("interrupted text", "ctx-int"):
            pass
        assert len(svc._pending_texts) == 1

        # Clear state (as _handle_interruption would)
        svc._clear_state()

        assert len(svc._pending_texts) == 0
        assert len(svc._current_audio_buffer) == 0
        assert len(svc._current_word_timestamps) == 0

    @pytest.mark.asyncio
    async def test_cache_stats(self, backend):
        """get_cache_stats returns correct hit/miss counts."""
        svc = self._make_service(backend)

        # One miss
        async for _ in svc.run_tts("stats test", "c1"):
            pass

        stats = await svc.get_cache_stats()
        assert stats["enabled"]
        assert stats["misses"] == 1
        assert stats["hits"] == 0
        assert stats["total_requests"] == 1
        assert stats["hit_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_clear_cache(self, backend):
        """clear_cache delegates to backend."""
        svc = self._make_service(backend)

        resp = CachedTTSResponse(
            audio_chunks=[CachedAudioChunk(_FAKE_AUDIO, 16000, 1)],
            sample_rate=16000,
            num_channels=1,
        )
        await backend.set("manual-key", resp)
        assert await backend.exists("manual-key")

        cleared = await svc.clear_cache()
        assert cleared == 1
        assert not await backend.exists("manual-key")

    @pytest.mark.asyncio
    async def test_different_text_different_cache(self, backend):
        """Different texts should not share cache entries."""
        svc = self._make_service(backend)

        async for f in svc.run_tts("text one", "c1"):
            if isinstance(f, TTSAudioRawFrame):
                await svc.push_frame(f, FrameDirection.DOWNSTREAM)
        await svc.push_frame(TTSStoppedFrame(context_id="c1"), FrameDirection.DOWNSTREAM)

        async for f in svc.run_tts("text two", "c2"):
            if isinstance(f, TTSAudioRawFrame):
                await svc.push_frame(f, FrameDirection.DOWNSTREAM)
        await svc.push_frame(TTSStoppedFrame(context_id="c2"), FrameDirection.DOWNSTREAM)

        assert svc._cache_misses == 2
        assert svc._cache_hits == 0
        assert len(svc.run_tts_calls) == 2

    @pytest.mark.asyncio
    async def test_cache_disabled_passthrough(self):
        """With cache disabled, run_tts passes through to parent unchanged."""
        svc = CachedMockTTS(cache_backend=None)
        frames = []
        async for f in svc.run_tts("passthrough", "ctx"):
            frames.append(f)

        assert len(svc.run_tts_calls) == 1
        assert svc.run_tts_calls[0] == ("passthrough", "ctx")
        assert any(isinstance(f, TTSAudioRawFrame) for f in frames)

    @pytest.mark.asyncio
    async def test_replayed_frames_marked_as_from_cache(self, backend):
        """Replayed frames should have the _tts_cache_origin attribute set."""
        svc = self._make_service(backend)

        # Populate cache
        async for f in svc.run_tts("mark test", "c1"):
            if isinstance(f, TTSAudioRawFrame):
                await svc.push_frame(f, FrameDirection.DOWNSTREAM)
        await svc.push_frame(TTSStoppedFrame(context_id="c1"), FrameDirection.DOWNSTREAM)

        # Replay
        replayed = []
        async for f in svc.run_tts("mark test", "c2"):
            replayed.append(f)

        assert len(replayed) > 0
        for f in replayed:
            assert svc._is_from_cache(f), f"{type(f).__name__} should be marked as from cache"

    @pytest.mark.asyncio
    async def test_push_frame_does_not_intercept_cache_replayed_frames(self, backend):
        """When replayed frames pass through push_frame, they should not be re-buffered."""
        svc = self._make_service(backend)

        # Populate cache
        async for f in svc.run_tts("no re-buffer", "c1"):
            if isinstance(f, TTSAudioRawFrame):
                await svc.push_frame(f, FrameDirection.DOWNSTREAM)
        await svc.push_frame(TTSStoppedFrame(context_id="c1"), FrameDirection.DOWNSTREAM)

        assert svc._cache_misses == 1

        # Replay
        replayed = []
        async for f in svc.run_tts("no re-buffer", "c2"):
            replayed.append(f)

        # No batch state should remain after replay
        assert len(svc._pending_texts) == 0
        assert len(svc._current_audio_buffer) == 0


# ---------------------------------------------------------------------------
# Tests: Cache key from settings
# ---------------------------------------------------------------------------


class TestCacheKeyFromSettings:
    """Test that _generate_cache_key reads from _settings correctly."""

    @pytest.fixture
    def backend(self):
        return MemoryCacheBackend(max_size=100)

    @pytest.mark.asyncio
    async def test_key_uses_settings_voice_and_model(self, backend):
        """Cache key should incorporate _settings.voice and _settings.model."""
        svc = CachedMockTTS(cache_backend=backend)
        key = svc._generate_cache_key("hello")
        assert isinstance(key, str)
        assert len(key) == 64

    @pytest.mark.asyncio
    async def test_different_voice_different_key(self, backend):
        """Changing the voice in settings should produce a different cache key."""
        svc1 = CachedMockTTS(cache_backend=backend)
        svc2 = CachedMockTTS(cache_backend=backend)

        svc1._settings.voice = "alice"
        svc2._settings.voice = "bob"

        key1 = svc1._generate_cache_key("hello")
        key2 = svc2._generate_cache_key("hello")
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_same_settings_same_key(self, backend):
        """Two services with identical settings should produce the same cache key."""
        svc1 = CachedMockTTS(cache_backend=backend)
        svc2 = CachedMockTTS(cache_backend=backend)

        svc1._settings.voice = "alice"
        svc2._settings.voice = "alice"

        assert svc1._generate_cache_key("hello") == svc2._generate_cache_key("hello")


# ---------------------------------------------------------------------------
# Tests: Word timestamps
# ---------------------------------------------------------------------------


class TestWordTimestampCaching:
    """Test caching interaction with word timestamps."""

    @pytest.fixture
    def backend(self):
        return MemoryCacheBackend(max_size=100)

    @pytest.mark.asyncio
    async def test_word_timestamps_collected_in_batch(self, backend):
        """add_word_timestamps should collect timestamps when batch is active."""
        svc = CachedMockTTS(cache_backend=backend)

        # Start a run_tts to create batch state
        async for _ in svc.run_tts("hello world", "ctx"):
            pass
        assert len(svc._pending_texts) == 1

        # Add word timestamps
        await svc.add_word_timestamps([("hello", 0.0), ("world", 0.5)], context_id="ctx")
        assert len(svc._current_word_timestamps) == 2
        assert svc._current_word_timestamps[0] == ("hello", 0.0)
        assert svc._current_word_timestamps[1] == ("world", 0.5)

    @pytest.mark.asyncio
    async def test_sentinel_timestamps_filtered(self, backend):
        """TTSStoppedFrame and Reset sentinel words should be filtered from collection."""
        svc = CachedMockTTS(cache_backend=backend)

        # Create batch state manually
        svc._pending_texts.append("hello")
        svc._current_audio_buffer.append(
            CachedAudioChunk(audio=_FAKE_AUDIO, sample_rate=16000, num_channels=1)
        )

        # Call add_word_timestamps with only real words (no sentinels)
        await svc.add_word_timestamps([("hello", 0.0)], context_id="ctx")
        assert len(svc._current_word_timestamps) == 1
        assert svc._current_word_timestamps[0][0] == "hello"

        # Verify sentinels would be filtered if present alongside real words
        svc._current_word_timestamps.clear()
        word_times = [("world", 0.1), ("TTSStoppedFrame", 0.0), ("Reset", 0.0)]
        filtered = [(w, t) for w, t in word_times if w not in ("TTSStoppedFrame", "Reset")]
        assert len(filtered) == 1
        assert filtered[0][0] == "world"

    @pytest.mark.asyncio
    async def test_timestamps_not_collected_without_batch(self, backend):
        """When no batch is active, timestamps should not be collected."""
        svc = CachedMockTTS(cache_backend=backend)
        await svc.add_word_timestamps([("hello", 0.0)], context_id="ctx")
        assert len(svc._current_word_timestamps) == 0


# ---------------------------------------------------------------------------
# Tests: Batch finalization
# ---------------------------------------------------------------------------


class TestBatchFinalization:
    """Test cache storage triggered by TTSStoppedFrame via push_frame."""

    @pytest.fixture
    def backend(self):
        return MemoryCacheBackend(max_size=100)

    @pytest.mark.asyncio
    async def test_single_task_cached_on_stop_frame(self, backend):
        """A single-sentence TTS should be cached when TTSStoppedFrame is pushed."""
        svc = CachedMockTTS(cache_backend=backend)

        # Generate audio (cache miss)
        frames = []
        async for f in svc.run_tts("cache me", "c1"):
            frames.append(f)

        # Push audio and stop frames to trigger caching
        for f in frames:
            if isinstance(f, TTSAudioRawFrame):
                await svc.push_frame(f, FrameDirection.DOWNSTREAM)
        await svc.push_frame(TTSStoppedFrame(context_id="c1"), FrameDirection.DOWNSTREAM)

        # Batch state should be cleared
        assert len(svc._pending_texts) == 0
        assert len(svc._current_audio_buffer) == 0

        # Backend should have the entry
        stats = await backend.get_stats()
        assert stats["size"] == 1

    @pytest.mark.asyncio
    async def test_no_audio_defers_then_discards(self, backend):
        """If no audio arrived, TTSStoppedFrame defers; next run_tts discards."""
        svc = CachedMockTTS(cache_backend=backend)

        # run_tts without pushing any audio frames
        async for _ in svc.run_tts("empty", "c1"):
            pass
        # Stop frame with no audio → deferred
        await svc.push_frame(TTSStoppedFrame(context_id="c1"), FrameDirection.DOWNSTREAM)

        assert svc._deferred is True
        assert len(svc._pending_texts) == 1  # still pending

        # Next run_tts resolves deferred state — discards since no audio came
        async for _ in svc.run_tts("next", "c2"):
            pass
        assert svc._deferred is False
        assert "empty" not in svc._pending_texts
        stats = await backend.get_stats()
        assert stats["size"] == 0

    @pytest.mark.asyncio
    async def test_silence_frames_buffered_not_used_as_delimiter(self, backend):
        """Silence frames are buffered as audio, not used as store triggers."""
        svc = CachedMockTTS(cache_backend=backend)

        async for f in svc.run_tts("with silence.", "c1"):
            if isinstance(f, TTSAudioRawFrame):
                await svc.push_frame(f, FrameDirection.DOWNSTREAM)

        # Silence frame is just another audio frame — no STORE yet
        silence_audio = b"\x00" * 640
        silence = TTSAudioRawFrame(
            audio=silence_audio, sample_rate=_SAMPLE_RATE, num_channels=1,
        )
        silence.metadata["_tts_silence"] = True
        await svc.push_frame(silence, FrameDirection.DOWNSTREAM)

        # Not stored yet — still pending
        assert len(svc._pending_texts) == 1
        assert len(svc._current_audio_buffer) == 2

        # TTSStoppedFrame triggers STORE (includes silence in cached audio)
        await svc.push_frame(TTSStoppedFrame(context_id="c1"), FrameDirection.DOWNSTREAM)

        stats = await backend.get_stats()
        assert stats["size"] == 1

        key = svc._generate_cache_key("with silence.")
        cached = await backend.get(key)
        assert cached is not None
        assert cached.audio_chunks[0].audio == _FAKE_AUDIO + silence_audio


# ---------------------------------------------------------------------------
# Tests: Cache hit behavior (serving_from_cache, metrics, frame types)
# ---------------------------------------------------------------------------


class TestCacheHitBehavior:
    """Test cache-hit-specific behavior."""

    @pytest.fixture
    def backend(self):
        return MemoryCacheBackend(max_size=100)

    @pytest.mark.asyncio
    async def test_serving_from_cache_flag_set_during_hit(self, backend):
        """_serving_from_cache should be True during cache replay, False otherwise."""
        svc = CachedMockTTS(cache_backend=backend)
        assert not svc._serving_from_cache

        # Populate cache
        async for f in svc.run_tts("flag test", "c1"):
            if isinstance(f, TTSAudioRawFrame):
                await svc.push_frame(f, FrameDirection.DOWNSTREAM)
        await svc.push_frame(TTSStoppedFrame(context_id="c1"), FrameDirection.DOWNSTREAM)
        assert not svc._serving_from_cache

        # During cache hit, the flag should be set (we check after)
        async for f in svc.run_tts("flag test", "c2"):
            pass
        # Flag should be cleared after run_tts returns
        assert not svc._serving_from_cache

    @pytest.mark.asyncio
    async def test_usage_metrics_skipped_on_cache_hit(self, backend):
        """start_tts_usage_metrics should not fire on cache hits."""
        svc = CachedMockTTS(cache_backend=backend)

        usage_calls: List[str] = []
        original_super_usage = TTSService.start_tts_usage_metrics

        async def tracking_usage(self_inner, text):
            usage_calls.append(text)
            await original_super_usage(self_inner, text)

        # Monkey-patch TTSService to track usage metric calls
        TTSService.start_tts_usage_metrics = tracking_usage

        try:
            # Populate cache (miss → usage metrics should fire from parent service)
            async for f in svc.run_tts("metrics test", "c1"):
                if isinstance(f, TTSAudioRawFrame):
                    await svc.push_frame(f, FrameDirection.DOWNSTREAM)
            await svc.push_frame(TTSStoppedFrame(context_id="c1"), FrameDirection.DOWNSTREAM)

            initial_count = len(usage_calls)

            # Cache hit — usage metrics should be skipped by the mixin
            async for _ in svc.run_tts("metrics test", "c2"):
                pass

            assert len(usage_calls) == initial_count, (
                f"Usage metrics should not fire on cache hit, but got "
                f"{len(usage_calls) - initial_count} extra calls"
            )
        finally:
            TTSService.start_tts_usage_metrics = original_super_usage

    @pytest.mark.asyncio
    async def test_no_duplicate_started_stopped_frames_on_hit(self, backend):
        """Cache replay should NOT yield TTSStartedFrame or TTSStoppedFrame.

        The base class manages these (push_start_frame, stop-frame handler).
        Yielding them from cache would cause duplicates.
        """
        svc = CachedMockTTS(cache_backend=backend)

        # Populate
        async for f in svc.run_tts("dup test", "c1"):
            if isinstance(f, TTSAudioRawFrame):
                await svc.push_frame(f, FrameDirection.DOWNSTREAM)
        await svc.push_frame(TTSStoppedFrame(context_id="c1"), FrameDirection.DOWNSTREAM)

        # Replay
        replayed = []
        async for f in svc.run_tts("dup test", "c2"):
            replayed.append(f)

        started = [f for f in replayed if isinstance(f, TTSStartedFrame)]
        stopped = [f for f in replayed if isinstance(f, TTSStoppedFrame)]
        assert len(started) == 0, f"Expected 0 TTSStartedFrame from replay, got {len(started)}"
        assert len(stopped) == 0, f"Expected 0 TTSStoppedFrame from replay, got {len(stopped)}"

    @pytest.mark.asyncio
    async def test_serving_from_cache_cleared_on_exception(self, backend):
        """_serving_from_cache flag should be cleared even if replay raises."""
        svc = CachedMockTTS(cache_backend=backend)

        # Manually populate backend with a corrupt entry
        corrupt = CachedTTSResponse(
            audio_chunks=[],  # empty — won't cause replay to raise but let's test the flag
            sample_rate=16000,
            num_channels=1,
        )
        key = svc._generate_cache_key("exception test")
        await backend.set(key, corrupt)

        async for _ in svc.run_tts("exception test", "c1"):
            pass

        assert not svc._serving_from_cache


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.fixture
    def backend(self):
        return MemoryCacheBackend(max_size=100)

    @pytest.mark.asyncio
    async def test_cache_get_failure_falls_through(self, backend):
        """If cache backend.get raises, should fall through to parent run_tts."""
        svc = CachedMockTTS(cache_backend=backend)

        original_get = backend.get

        async def failing_get(key):
            raise RuntimeError("backend down")

        backend.get = failing_get

        frames = []
        async for f in svc.run_tts("fallback test", "ctx"):
            frames.append(f)

        assert len(svc.run_tts_calls) == 1
        assert any(isinstance(f, TTSAudioRawFrame) for f in frames)

        backend.get = original_get

    @pytest.mark.asyncio
    async def test_run_tts_exception_clears_batch(self, backend):
        """If parent run_tts raises, batch state should be cleared."""

        class FailingTTS(TTSService):
            def can_generate_metrics(self):
                return False

            async def run_tts(self, text, context_id):
                raise RuntimeError("TTS synthesis failed")
                yield  # noqa: unreachable - makes this an async generator

        class CachedFailing(TTSCacheMixin, FailingTTS):
            pass

        svc = CachedFailing(cache_backend=backend, sample_rate=16000)

        with pytest.raises(RuntimeError, match="TTS synthesis failed"):
            async for _ in svc.run_tts("fail", "ctx"):
                pass

        assert len(svc._pending_texts) == 0

    @pytest.mark.asyncio
    async def test_clear_cache_without_backend(self):
        """clear_cache with no backend should return 0."""
        svc = CachedMockTTS(cache_backend=None)
        result = await svc.clear_cache()
        assert result == 0

    @pytest.mark.asyncio
    async def test_stats_without_backend(self):
        """get_cache_stats with no backend should still return base stats."""
        svc = CachedMockTTS(cache_backend=None)
        stats = await svc.get_cache_stats()
        assert not stats["enabled"]
        assert stats["hits"] == 0
        assert stats["misses"] == 0


# ---------------------------------------------------------------------------
# Tests: Full pipeline integration (using pipecat test utilities)
# ---------------------------------------------------------------------------


class TestFullPipelineIntegration:
    """Integration tests that run the cached TTS through a real pipecat pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_cache_roundtrip(self):
        """Run CachedMockTTS through run_test to verify pipeline-level behavior."""
        from pipecat.tests.utils import run_test

        backend = MemoryCacheBackend(max_size=100)
        svc = CachedMockTTS(cache_backend=backend)

        # First run: cache miss → parent generates audio
        down_frames, _ = await run_test(
            svc,
            frames_to_send=[TTSSpeakFrame(text="pipeline test")],
        )

        # Should have TTS frames in output
        down_types = [type(f) for f in down_frames]
        assert TTSAudioRawFrame in down_types

        # Second run with same text: cache hit → replayed from cache
        svc2 = CachedMockTTS(cache_backend=backend)
        down_frames2, _ = await run_test(
            svc2,
            frames_to_send=[TTSSpeakFrame(text="pipeline test")],
        )

        down_types2 = [type(f) for f in down_frames2]
        assert TTSAudioRawFrame in down_types2
        # The second service should have a cache hit
        assert svc2._cache_hits >= 1
