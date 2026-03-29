#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Tests for TTL-on-access (refresh on hit) behavior in cache backends."""

import asyncio
import time
from unittest.mock import patch

import pytest

from pipecat_tts_cache.backends.memory import MemoryCacheBackend
from pipecat_tts_cache.models import CachedAudioChunk, CachedTTSResponse

_FAKE_AUDIO = b"\x00\x01" * 160


@pytest.fixture
def sample_response():
    return CachedTTSResponse(
        audio_chunks=[CachedAudioChunk(audio=_FAKE_AUDIO, sample_rate=16000, num_channels=1)],
        sample_rate=16000,
        num_channels=1,
        total_duration_s=0.02,
    )


class TestMemoryTTLRefresh:
    """Test TTL refresh behavior in MemoryCacheBackend."""

    @pytest.mark.asyncio
    async def test_ttl_refreshed_on_hit(self, sample_response):
        """A cache hit should reset the expiry to now + original_ttl."""
        backend = MemoryCacheBackend(max_size=10, refresh_ttl_on_hit=True)

        ttl = 100  # seconds
        await backend.set("key1", sample_response, ttl=ttl)

        # Record the initial expiry
        async with backend._lock:
            _, initial_expiry, stored_ttl = backend._cache["key1"]
        assert stored_ttl == ttl

        # Simulate time passing (50 seconds)
        with patch("pipecat_tts_cache.backends.memory.time") as mock_time:
            fake_now = time.time() + 50
            mock_time.time.return_value = fake_now

            result = await backend.get("key1")
            assert result is not None

            # Expiry should now be fake_now + original_ttl, not the old value
            async with backend._lock:
                _, new_expiry, _ = backend._cache["key1"]
            assert new_expiry == pytest.approx(fake_now + ttl, abs=1)
            assert new_expiry > initial_expiry

    @pytest.mark.asyncio
    async def test_ttl_not_refreshed_when_disabled(self, sample_response):
        """With refresh_ttl_on_hit=False, expiry stays at original value."""
        backend = MemoryCacheBackend(max_size=10, refresh_ttl_on_hit=False)

        ttl = 100
        await backend.set("key1", sample_response, ttl=ttl)

        async with backend._lock:
            _, initial_expiry, _ = backend._cache["key1"]

        # Hit — should NOT refresh
        result = await backend.get("key1")
        assert result is not None

        async with backend._lock:
            _, expiry_after_hit, _ = backend._cache["key1"]

        # Expiry unchanged
        assert expiry_after_hit == initial_expiry

    @pytest.mark.asyncio
    async def test_no_ttl_entry_not_refreshed(self, sample_response):
        """Entries stored without TTL (ttl=0) should not be modified on hit."""
        backend = MemoryCacheBackend(max_size=10, refresh_ttl_on_hit=True)

        await backend.set("key1", sample_response, ttl=0)

        async with backend._lock:
            _, expiry, stored_ttl = backend._cache["key1"]
        assert expiry == 0.0
        assert stored_ttl == 0

        # Hit — should not touch entries with no TTL
        result = await backend.get("key1")
        assert result is not None

        async with backend._lock:
            _, expiry_after, _ = backend._cache["key1"]
        assert expiry_after == 0.0

    @pytest.mark.asyncio
    async def test_expired_entry_not_returned(self, sample_response):
        """Even with refresh enabled, expired entries must not be returned."""
        backend = MemoryCacheBackend(max_size=10, refresh_ttl_on_hit=True)

        await backend.set("key1", sample_response, ttl=10)

        # Simulate time well past expiry
        with patch("pipecat_tts_cache.backends.memory.time") as mock_time:
            mock_time.time.return_value = time.time() + 100
            result = await backend.get("key1")
            assert result is None

    @pytest.mark.asyncio
    async def test_refresh_is_default_enabled(self, sample_response):
        """refresh_ttl_on_hit should default to True."""
        backend = MemoryCacheBackend(max_size=10)
        assert backend._refresh_ttl_on_hit is True

    @pytest.mark.asyncio
    async def test_multiple_hits_keep_extending(self, sample_response):
        """Repeated hits should keep pushing the expiry forward."""
        backend = MemoryCacheBackend(max_size=10, refresh_ttl_on_hit=True)

        ttl = 60
        await backend.set("key1", sample_response, ttl=ttl)

        base_time = time.time()

        for i in range(5):
            with patch("pipecat_tts_cache.backends.memory.time") as mock_time:
                fake_now = base_time + (i + 1) * 30  # 30s, 60s, 90s, 120s, 150s
                mock_time.time.return_value = fake_now

                result = await backend.get("key1")
                # Each hit at +30s intervals with TTL=60: always within TTL
                assert result is not None

                async with backend._lock:
                    _, expiry, _ = backend._cache["key1"]
                # Expiry should be fake_now + original_ttl
                assert expiry == pytest.approx(fake_now + ttl, abs=1)
