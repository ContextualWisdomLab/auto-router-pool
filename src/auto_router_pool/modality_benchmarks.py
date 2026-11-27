"""Public benchmark scores for non-chat model families.

A score is only compared inside its own kind: image and video use Elo, audio
uses ``100 - AA-WER`` so higher is better, and OCR has no public source yet.
The values are pinned with the URL they came from; an operator can override
them without a code change in ``<home>/auto-pool-modality-benchmarks.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

SOURCES = {
    "image": "https://artificialanalysis.ai/image/leaderboard/text-to-image",
    "video": "https://artificialanalysis.ai/video/leaderboard/text-to-video",
    "audio": "https://artificialanalysis.ai/speech-to-text/non-streaming",
    "ocr": "",
}

# Pinned 2026-11-27. Audio scores are 100 minus the AA-WER percentage, so the
# same higher-is-better comparison works for every kind.
DEFAULTS = {
    "image": {
        "gpt-image-2.5-sunburst": 1230.0,
        "gpt-image-2.5-flare": 1212.0,
        "gpt-image-2": 1188.0,
    },
    "video": {
        "grok-imagine-video": 1062.0,
        "grok-imagine-video-1.5": 1051.0,
    },
    "audio": {
        "gpt-transcribe": 96.7,
        "gpt-4o-transcribe": 96.0,
        "gpt-4o-mini-transcribe": 95.5,
    },
    "ocr": {},
}


def load(home: Path | None = None) -> dict[str, dict[str, float]]:
    """Pinned scores merged with an optional operator-maintained file."""
    scores = {kind: dict(entries) for kind, entries in DEFAULTS.items()}
    if home is None:
        return scores
    path = Path(home) / "auto-pool-modality-benchmarks.json"
    if not path.exists():
        return scores
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return scores
    if not isinstance(payload, dict):
        return scores
    for kind, entries in payload.items():
        if kind not in scores or not isinstance(entries, dict):
            continue
        for name, value in entries.items():
            if isinstance(value, dict):
                value = value.get("score")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                scores[kind][str(name)] = float(value)
    return scores
