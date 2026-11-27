"""Tier templates for the three auto-router models.

Effort assignments come from the Artificial Analysis effort ladder, not from the
tier name: a small model can lose about two thirds of its index at ``low``, so
the light tiers run it at ``medium``/``high``. The frontier tier is instead
placed for equal score: its entries are fallbacks for the same request, so it
targets the weakest ceiling in the chain instead of each model's own peak.
"""
from __future__ import annotations

from .assign import SPREAD, place

ROUTERS = ("auto", "auto-classify", "auto-strix")

# Tier names are Korean because they are the router's published contract; the
# classifier prompt refers to them by name.
AUTO = {
    "명시 정보 추출·단순 분류·고정 규칙 처리": "simple",
    "문맥 기반 분류·관계 추출·의도별 라우팅": "context",
    "시각 자료 해석·화면 조작·글쓰기": "visual",
    "일상적인 코딩·데이터 분석·도구 사용": "daily",
    "복잡한 코딩·시스템 설계·기업 지식 업무": "complex",
    "고난도 과제·심화 검토": "review",
    "최고난도 추론·심층 연구·복잡한 장기 작업": "frontier",
}


def top_effort(variants: dict) -> str:
    """Effort where this model measured its highest index.

    ``max`` is not always above ``xhigh`` (claude-fable-5-1 loses 0.9 at max),
    so the top tier reads the ladder instead of assuming a label.
    """
    ranked = [(entry.get("index") or 0, effort) for effort, entry in variants.items()]
    return max(ranked)[1] if ranked else "xhigh"


def _compact(tier_map: dict) -> dict:
    """Drop padded or duplicate entries while preserving routing order."""
    cleaned = {}
    for tier, entries in tier_map.items():
        seen, keep = set(), []
        for model, effort in entries:
            if not model or model in seen:
                continue
            seen.add(model)
            keep.append((model, effort))
        cleaned[tier] = keep
    return cleaned


def _frontier(premium, primary, runner_up, top, ladders, band):
    """Top tier entries: score-equal when every model has a measured ladder."""
    fallback = [
        (premium, top.get(premium, "xhigh")),
        (primary, top.get(primary, "xhigh")),
        (runner_up, top.get(runner_up, "xhigh")),
    ]
    if not ladders:
        return fallback
    placed = place([premium, primary, runner_up], ladders, band)
    if len(placed["entries"]) != 3:
        return fallback
    return [(model, effort) for model, effort, _ in placed["entries"]]


def auto_tiers(
    primary: str,
    premium: str,
    budget: str,
    runner_up: str,
    micro=(),
    top=None,
    ladders=None,
    band: float = SPREAD,
):
    micro = list(micro)[:2] + [""] * (2 - len(list(micro)[:2]))
    top = top or {}
    return _compact({
        "명시 정보 추출·단순 분류·고정 규칙 처리": [
            (budget, "medium"), (micro[0], "low"), (micro[1], "low"), (primary, "low")],
        "문맥 기반 분류·관계 추출·의도별 라우팅": [
            (budget, "high"), (micro[0], "medium"), (micro[1], "medium"), (primary, "medium")],
        "시각 자료 해석·화면 조작·글쓰기": [
            (primary, "medium"), (premium, "medium")],
        "일상적인 코딩·데이터 분석·도구 사용": [
            (primary, "medium"), (runner_up, "medium"), (premium, "medium")],
        "복잡한 코딩·시스템 설계·기업 지식 업무": [
            (primary, "high"), (premium, "high"), (runner_up, "high")],
        "고난도 과제·심화 검토": [
            (primary, "high"), (premium, "high"), (runner_up, "high")],
        "최고난도 추론·심층 연구·복잡한 장기 작업": _frontier(
            premium, primary, runner_up, top, ladders, band),
    })


def classify_tiers(primary: str, premium: str, budget: str, runner_up: str, micro=()):
    micro = list(micro)[:2] + [""] * (2 - len(list(micro)[:2]))
    return _compact({
        "SIMPLE": [(budget, "medium"), (micro[0], "low"), (micro[1], "low"), (primary, "low")],
        "MEDIUM": [(budget, "high"), (micro[0], "medium"), (micro[1], "medium"), (primary, "medium")],
        "COMPLEX": [(primary, "low"), (runner_up, "low"), (premium, "low")],
        "REASONING": [(primary, "medium"), (runner_up, "medium"), (premium, "medium")],
    })


def strix_tiers(primary: str, premium: str, budget: str, runner_up: str, micro=(), top=None):
    micro = list(micro)[:2] + [""] * (2 - len(list(micro)[:2]))
    top = top or {}
    return _compact({
        "SIMPLE": [(budget, "medium"), (micro[0], "low"), (micro[1], "low")],
        "MEDIUM": [(primary, "medium")],
        "COMPLEX": [(primary, "high")],
        "REASONING": [(primary, top.get(primary, "xhigh"))],
    })


def routers(
    primary: str,
    premium: str,
    budget: str,
    runner_up: str,
    micro=(),
    top=None,
    ladders=None,
    band: float = SPREAD,
):
    """Map router name -> tier spec.

    ``top`` maps a model to its highest measured effort, used where a tier holds
    a single model. ``ladders`` maps a model to its full measured ladder so the
    frontier tier can be placed for equal score instead.
    """
    return {
        "auto": auto_tiers(
            primary, premium, budget, runner_up, micro, top, ladders, band
        ),
        "auto-classify": classify_tiers(primary, premium, budget, runner_up, micro),
        "auto-strix": strix_tiers(primary, premium, budget, runner_up, micro, top),
    }


def build(tier_map: dict) -> tuple[dict, dict]:
    """``(tiers, tier_model_configs)`` exactly as LiteLLM stores them."""
    tiers = {name: [model for model, _ in entries] for name, entries in tier_map.items()}
    configs = {
        name: [
            {"model_name": model, "litellm_params": {"reasoning_effort": effort}}
            for model, effort in entries
        ]
        for name, entries in tier_map.items()
    }
    return tiers, configs
