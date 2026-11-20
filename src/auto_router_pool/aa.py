"""Artificial Analysis Intelligence Index, parsed from the model page payload.

The page embeds a ``initialModels`` JSON array with one entry per effort variant
(``name``, ``name-high``, ...), each carrying ``intelligenceIndex`` and per-million
prices.  Parsing is deliberately strict about *structural*
failure: a renamed field or a client-side render must surface as an error, never
as an empty result, otherwise a newly registered model would be dropped silently.
"""
from __future__ import annotations

import json
import time
import urllib.request

DEFAULT_BASE = "https://artificialanalysis.ai/models/"
DEFAULT_TTL = 7 * 24 * 3600
MARKERS = ('"initialModels":[', '"models":[')
STRUCTURAL = ("missing_array", "fetch_error")


class LayoutError(ValueError):
    """The page no longer contains a readable model array."""


def parse_variants(html: str, slug: str) -> dict[str, dict]:
    """Extract ``{effort: {index, in, out}}`` for one model slug."""
    text = html.replace("\\", "")
    for marker in MARKERS:
        pos = text.find(marker)
        if pos >= 0:
            break
    else:
        raise LayoutError("missing_array")

    start = pos + text[pos:].index(":[") + 1  # land on '['
    models, _ = json.JSONDecoder().raw_decode(text[start:])

    variants: dict[str, dict] = {}
    for model in models:
        name = model.get("slug") or ""
        if not (name == slug or name.startswith(slug + "-")):
            continue
        index = model.get("intelligenceIndex")
        if index is None:
            continue
        effort = (model.get("effort") or {}).get("slug") or (
            "max" if name == slug else name[len(slug) + 1 :]
        )
        variants[effort] = {
            "index": index,
            "in": model.get("price1mInputTokens"),
            "out": model.get("price1mOutputTokens"),
        }
    return variants


def fetch_page(url: str, timeout: float = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "auto-router-pool/1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def get_variants(
    slug: str,
    cache: dict,
    *,
    base: str = DEFAULT_BASE,
    ttl: float = DEFAULT_TTL,
    force: bool = False,
    fetch=fetch_page,
):
    """Return ``(variants, error)``; cached numbers are reused inside ``ttl``."""
    entry = cache.get(slug)
    if not force and entry and time.time() - entry["fetched"] < ttl:
        return entry["variants"], None

    try:
        variants = parse_variants(fetch(base + slug), slug)
    except LayoutError:
        return (entry["variants"] if entry else {}), "missing_array"
    except Exception:
        return (entry["variants"] if entry else {}), "fetch_error"

    if not variants:
        # Parsed fine, but this model is not in the comparison set.
        return {}, "no_aa_entry"
    cache[slug] = {"fetched": time.time(), "variants": variants}
    return variants, None
