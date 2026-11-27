"""Artificial Analysis Intelligence Index, parsed from the model page payload.

The page embeds a ``initialModels`` JSON array with one entry per effort variant
(``name``, ``name-high``, ...), each carrying ``intelligenceIndex`` and per-million
prices.  Parsing is deliberately strict about *structural*
failure: a renamed field or a client-side render must surface as an error, never
as an empty result, otherwise a newly registered model would be dropped silently.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "https://artificialanalysis.ai/models/"
DECISION_BASE = "https://artificialanalysis.ai/decision-models/"
DEFAULT_TTL = 7 * 24 * 3600
# A model AA does not score yet can gain a page later; keep a 404 for a day so a
# transient miss or a new release is picked up without refetching every run.
NEGATIVE_TTL = 24 * 3600
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


# The Decision Index uses its own model pages and slugs. Gateway names are not
# always the AA slug, and a free tier is the same model as its base.
DECISION_ALIASES = {
    "gpt-6-luna": "gpt-6-luna-decisions",
    "jev": "jev-1-14",
    "jev-latest": "jev-1-14",
}
_DECISION_SCORE = re.compile(
    r"scores ([\d.]+) on the Artificial Analysis Decision Index"
)
_DECISION_COST = re.compile(r"costs \$([\d,.]+) per 1M tasks")
_DECISION_TIME = re.compile(r"At ([\d.]+) ms per task")
_DECISION_PRICE = re.compile(
    r"Pricing for .*? is \$([\d.]+) per 1M input tokens"
)


def decision_slug(name: str, aliases: dict | None = None) -> str:
    """AA's Decision Index slug for a gateway model name.

    ``aliases`` maps a gateway model to the base model whose AA page should be
    borrowed when the model has none of its own. The alias is applied before the
    built-in ``DECISION_ALIASES`` table, so a correction can target any scored
    base model rather than only the hardcoded ones.
    """
    base = str(name).removesuffix(":free")
    if aliases:
        base = str(aliases.get(base, base)).removesuffix(":free")
    return DECISION_ALIASES.get(base, base)


def _number(pattern, text: str) -> float | None:
    match = pattern.search(text)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def parse_decision_page(html: str) -> dict:
    """Decision Index score, cost and time from one AA model page.

    AA renders these as prose (``scores 60 on ...``), not inside the comparison
    array, so the prose is the stable part to read. A page that renders only on
    the client raises ``missing_score`` instead of silently scoring zero.
    """
    score = _number(_DECISION_SCORE, html)
    if score is None:
        raise LayoutError("missing_score")
    return {
        "score": score,
        "cost_per_million_tasks": _number(_DECISION_COST, html),
        "time_per_task_ms": _number(_DECISION_TIME, html),
        "input_price_per_million": _number(_DECISION_PRICE, html),
    }


def get_decision_scores(
    names,
    cache: dict,
    *,
    base: str = DECISION_BASE,
    ttl: float = DEFAULT_TTL,
    force: bool = False,
    fetch=None,
    aliases: dict | None = None,
) -> tuple[dict[str, dict], dict[str, str]]:
    """``({gateway name: entry}, {name: reason})`` from AA's Decision Index.

    Scores are cached by AA slug for the same week as the Intelligence Index. A
    404 is a model AA does not score; a layout break keeps any stale score
    rather than demoting the model to unmeasured on a transient page change.
    """
    fetch = fetch or fetch_page
    found: dict[str, dict] = {}
    errors: dict[str, str] = {}
    now = time.time()
    for name in dict.fromkeys(names):
        slug = decision_slug(name, aliases)
        entry = cache.get(slug)
        entry_ttl = NEGATIVE_TTL if entry and entry.get("score") is None else ttl
        if not force and entry and now - entry.get("fetched", 0) < entry_ttl:
            if entry.get("score") is not None:
                found[name] = entry
            elif entry.get("error"):
                errors[name] = entry["error"]
            continue
        try:
            parsed = parse_decision_page(fetch(base + slug))
        except LayoutError as error:
            if entry and entry.get("score") is not None:
                found[name] = entry
            errors[name] = str(error)
            continue
        except urllib.error.HTTPError as error:
            if error.code == 404:
                cache[slug] = {
                    "fetched": now, "score": None, "error": "no_aa_entry",
                }
                errors[name] = "no_aa_entry"
            else:
                errors[name] = "fetch_error"
            continue
        except Exception:
            errors[name] = "fetch_error"
            continue
        cache[slug] = {"fetched": now, **parsed}
        found[name] = cache[slug]
    return found, errors


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
    except urllib.error.HTTPError as error:
        # A 404 is the page saying "no such model", which is a skip, not a
        # layout break; 5xx and 429 stay structural so they are retried.
        if error.code == 404:
            return (entry["variants"] if entry else {}), "no_aa_entry"
        return (entry["variants"] if entry else {}), "fetch_error"
    except Exception:
        return (entry["variants"] if entry else {}), "fetch_error"

    if not variants:
        # Parsed fine, but this model is not in the comparison set.
        return {}, "no_aa_entry"
    cache[slug] = {"fetched": time.time(), "variants": variants}
    return variants, None
