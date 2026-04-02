#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Cache key generation for TTS caching."""

import hashlib
import json


def normalize_text(text: str) -> str:
    """Normalize text for consistent cache key generation."""
    return " ".join(text.strip().split())


def generate_cache_key(
    text: str,
    voice_id: str,
) -> str:
    """Generate a deterministic SHA-256 cache key from text and voice_id."""
    normalized_text = normalize_text(text)
    if not normalized_text:
        raise ValueError("Cannot generate cache key for empty text")

    key_data = {
        "text": normalized_text,
        "voice_id": voice_id,
    }

    key_string = json.dumps(key_data, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(key_string.encode("utf-8")).hexdigest()
