"""Tier templates for the managed routers.

LiteLLM caps a router at eight published tiers and rejects a tier map that does
not cover every definition, so the taxonomy is built to fit that budget exactly
and to be written atomically with its classifier contract.

``auto`` -- the task partition, disjoint by *what the request must produce*:

* structured text (one-off extraction and rule application);
* free-form text (conversation, summary, translation, creative writing) -- also
  the catch-all, so nothing is left unclassified;
* code and data, split by scope (one change versus system-wide design);
* long-horizon reasoning and hard verification;
* image or screen *input* with a text answer;
* image *output*;
* video *output*.

Precedence is fixed: output modality first, then media input, then code, then
text shape, with ``대화·요약·번역·창작 텍스트`` as the default. A tier with no model
is dropped instead of published empty, and the fallback tier is the catch-all.

``auto-strix`` -- security work: static analysis, exploit reproduction, threat
modelling, incident response.
``auto-advisor`` -- model behaviour monitoring: drift, regressions, root cause,
guardrail review.

LiteLLM registers **one** router with operator-defined tiers or an
operator-written classifier prompt, and that budget belongs to ``auto``. The two
focus routers therefore keep the shipped four tiers (``SIMPLE`` .. ``REASONING``)
and are separated by *which models* they hand each tier: security work is priced
and staffed for code reading at every level, monitoring starts on the cheapest
model that still answers and climbs only when a case gets hard.

Effort assignments come from the Artificial Analysis ladder, not from the tier
name. The top tier of each router is placed for equal score, because its entries
are fallbacks for the same request.
"""
from __future__ import annotations

from .assign import SPREAD, grow, place

ROUTERS = ("auto", "auto-classify", "auto-strix", "auto-advisor")

# LiteLLM refuses a router whose tier_definitions is outside this range.
MIN_TIERS = 2
MAX_TIERS = 8

T_SIMPLE = "명시 정보 추출·단순 분류·고정 규칙 처리"
T_CHAT = "대화·요약·번역·창작 텍스트"
T_CODE = "코드·데이터·도구 사용"
T_OPS = "복잡한 코딩·시스템 설계·기업 지식 업무"
FRONTIER = "최고난도 추론·심층 연구·복잡한 장기 작업"
T_VISION = "이미지·화면 해석·화면 조작"
T_IMAGE = "이미지 생성·편집"
T_VIDEO = "비디오 생성·편집"

MEDIA_TIERS = (T_IMAGE, T_VIDEO)

# The shipped tiers, used by the routers that may not define their own.
SHIPPED = ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING")

# Adaptive per turn: classify every request, including the tool-result
# continuations of the same turn, so a turn that turns out harder escalates
# mid-loop. The engine treats 'user_turn' and session affinity as the opposite
# choice -- both replay a held decision and never see the later tool calls.
POLICY = {"classification_mode": "every_request", "session_affinity": False}

# Published contract: the classifier reads these names, so they travel with the
# tier map instead of being inferred from code.
AUTO = {
    T_SIMPLE: "simple",
    T_CHAT: "chat",
    T_CODE: "code",
    T_OPS: "ops",
    FRONTIER: "frontier",
    T_VISION: "vision",
    T_IMAGE: "image",
    T_VIDEO: "video",
}

AUTO_DEFINITIONS = {
    T_SIMPLE: (
        "출력이 짧고 정답이 입력에 직접 드러나며, 숨은 의미 해석 없이 정해진 규칙만 "
        "적용하면 끝나는 작업입니다. 지정 필드 추출, 라벨 변환, 형식 정규화, 상태 코드에 "
        "따른 분기처럼 값과 규칙이 명확할 때 배정합니다. 항목이 많다는 이유만으로 상위 "
        "tier를 고르지 않습니다."
    ),
    T_CHAT: (
        "코드·도구·이미지 없이 텍스트만으로 완결되는 대화·요약·번역·창작입니다. 난도가 "
        "문장 구성과 정확성에 있고, 외부 조사나 여러 시스템의 변경이 필요하지 않을 때 "
        "배정합니다. 다른 tier의 완료 조건에 해당하지 않는 요청의 기본 tier입니다."
    ),
    T_CODE: (
        "코드·데이터·도구를 다루되 범위가 하나의 변경·하나의 쿼리·정해진 절차에 머무는 "
        "작업입니다. 로컬 코드 수정, 표준적인 정제·집계, 일반적인 SQL, 절차가 정해진 도구 "
        "실행처럼 결과를 직접 확인할 수 있을 때 배정합니다. 여러 구성 요소의 설계를 바꾸거나 "
        "원인을 장기간 추적해야 하면 복잡한 코딩 tier로 보냅니다."
    ),
    T_OPS: (
        "목표는 분명하지만 여러 시스템·코드·문서·업무 규칙의 관계를 이해하고 대안의 "
        "장단점을 판단해야 하는 전문 작업입니다. 의존성, 변경 영향, 예외 조건, 운영 제약을 "
        "함께 검토해 설계·구현·원인 분석·의사결정을 수행할 때 배정합니다. 새로운 이론이나 "
        "해결 전략 자체를 찾아야 하면 최상위 tier로 보냅니다."
    ),
    FRONTIER: (
        "해결 경로가 명확하지 않거나, 범위는 정해졌어도 긴 추론 사슬과 정밀한 검증이 필요한 "
        "작업입니다. 여러 분야의 근거를 연결하고 경쟁 가설을 검토하며 전략을 수정하는 개방형 "
        "연구, 까다로운 수학·논리·알고리즘 정당성 검토, 기존 결론의 반례·누락·모순 검토를 "
        "포함합니다. 입력이 길거나 도구 호출이 많다는 이유만으로 선택하지 않습니다."
    ),
    T_VISION: (
        "입력에 이미지·화면·문서가 포함되어 그 내용과 배치를 이해하거나 화면 상태에 맞춰 "
        "조작하는 것이 완료 조건인 작업입니다. 산출물은 텍스트이며, 이미지가 첨부되었다는 "
        "이유만으로 선택하지 않습니다. 단순 필드 추출은 경량 tier로 보냅니다."
    ),
    T_IMAGE: (
        "산출물 자체가 이미지인 요청입니다. 텍스트·참조 이미지로부터 이미지를 생성하거나 "
        "편집하며, 다른 tier보다 먼저 판정합니다. 텍스트 답변이 함께 필요해도 산출물의 핵심이 "
        "이미지이면 이 tier로 분류합니다."
    ),
    T_VIDEO: (
        "산출물 자체가 비디오인 요청입니다. 텍스트·참조 이미지로부터 비디오를 생성하거나 "
        "편집하며, 다른 tier보다 먼저 판정합니다. 프레임 설명이나 대본 작성처럼 텍스트가 "
        "핵심이면 이 tier가 아니라 해당 텍스트 tier로 분류합니다."
    ),
}

AUTO_EXAMPLES = {
    T_SIMPLE: "주문서에서 명시된 주문번호·금액·날짜를 추출 → " + T_SIMPLE,
    T_CHAT: "주어진 회의록을 3문장으로 요약하거나 영문 안내문을 한국어로 번역 → " + T_CHAT,
    T_CODE: "정의된 지표의 SQL 집계를 작성하거나 원인이 확정된 국소 버그 수정 → " + T_CODE,
    T_OPS: "여러 서비스의 인증·권한·데이터 흐름을 검토해 장애 원인과 변경 영향 분석 → " + T_OPS,
    FRONTIER: "상충하는 근거를 조사하고 경쟁 가설을 검증하며 전략을 수정 → " + FRONTIER,
    T_VISION: "스크린샷의 레이아웃을 비교하거나 화면 지시에 따라 입력 폼을 조작 → " + T_VISION,
    T_IMAGE: "제품 사진을 배경이 투명한 광고 이미지로 편집 → " + T_IMAGE,
    T_VIDEO: "15초 제품 소개 영상을 생성 → " + T_VIDEO,
}

AUTO_PROMPT = (
    "현재 요청이 실제로 수행해야 하는 작업과, 이를 정확히 완료하는 데 남은 판단의 난도를 "
    "평가합니다. 다음 순서로 판정해 정확히 하나의 tier를 고릅니다. ① 산출물이 이미지면 "
    "이미지 생성·편집, ② 산출물이 비디오면 비디오 생성·편집, ③ 입력의 이미지·화면 해석이나 "
    "화면 조작이 완료 조건이면 이미지·화면 해석·화면 조작, ④ 코드·데이터·도구 작업이면 범위에 "
    "따라 코드·데이터·도구 사용 또는 복잡한 코딩·시스템 설계·기업 지식 업무, ⑤ 나머지 "
    "텍스트는 명시 정보 추출·단순 분류·고정 규칙 처리와 대화·요약·번역·창작 텍스트를 먼저 "
    "비교하고, 검증 부담이 크면 최고난도 추론·심층 연구·복잡한 장기 작업으로 보냅니다. 어느 "
    "tier에도 확신이 없으면 대화·요약·번역·창작 텍스트로 분류합니다. 요구 품질과 완료 조건을 "
    "충족하는 가장 낮은 tier를 고르되, 해결되지 않은 복잡한 판단을 비용 때문에 낮추지 "
    "않습니다. 이전 대화는 생략된 대상·제약·진행 상태를 해석하는 데 사용하고, '계속'·'네'·"
    "'진행해' 같은 짧은 요청은 이어서 수행할 작업의 난도로 분류합니다. 이전 작업이 어려웠더라도 "
    "현재 요청이 확정된 결과의 추출·정리·형식 변환이면 경량 tier를 고릅니다. 입력 길이, 항목 "
    "수, 이미지·문서의 존재, 도구 호출 수만으로 난도를 높이지 않습니다. tier 이름과 "
    "reasoning_effort를 기계적으로 일치시키지 않고 tier_model_configs의 모델별 설정을 사용합니다."
)

# Token mix each tier's requests actually send, used to price candidates from
# their input and output prices separately.
TIER_PROFILE = {
    T_SIMPLE: "extract",
    T_CHAT: "chat",
    T_CODE: "code",
    T_OPS: "ops",
    FRONTIER: "deep",
    T_VISION: "chat",
    T_IMAGE: "media",
    T_VIDEO: "media",
}

# The shipped tiers are priced per router: classification asks for one short
# answer, monitoring is high-volume extraction and reproduction, security work
# reads code and evidence on every tier.
CLASSIFY_PROFILE = {
    "SIMPLE": "extract", "MEDIUM": "chat", "COMPLEX": "code", "REASONING": "deep",
}
STRIX_PROFILE = {
    "SIMPLE": "code", "MEDIUM": "ops", "COMPLEX": "ops", "REASONING": "deep",
}
ADVISOR_PROFILE = {
    "SIMPLE": "extract", "MEDIUM": "code", "COMPLEX": "review", "REASONING": "deep",
}


def top_effort(variants: dict) -> str:
    """Effort where this model measured its highest index.

    ``max`` is not always above ``xhigh`` (claude-fable-5-1 loses 0.9 at max),
    so the top tier reads the ladder instead of assuming a label.
    """
    ranked = [(entry.get("index") or 0, effort) for effort, entry in variants.items()]
    return max(ranked)[1] if ranked else "xhigh"


def _compact(tier_map: dict) -> dict:
    """Drop padded, duplicate and empty entries, preserving routing order.

    A tier that ends up with no model is removed entirely: publishing an empty
    tier is the classification failure this template must avoid.
    """
    cleaned = {}
    for tier, entries in tier_map.items():
        seen, keep = set(), []
        for model, effort in entries:
            if not model or model in seen:
                continue
            seen.add(model)
            keep.append((model, effort))
        if keep:
            cleaned[tier] = keep
    return cleaned


def _top(entries, ladder_models, models_by_role, top, ladders, band, profile,
         free_models=()):
    """Top tier: equal-score placement when the ladders are known."""
    fallback = [(models_by_role[role], top.get(models_by_role[role], "xhigh")) for role in entries]
    if not ladders:
        return fallback
    placed = place(
        [name for name, _ in fallback], ladders, band, profile, free_models
    )
    if len(placed["entries"]) != len(fallback):
        return fallback
    return [(model, effort) for model, effort, _ in placed["entries"]]


def _frontier(premium, primary, runner_up, top, ladders, band, free_models=()):
    return _top(
        ("premium", "primary", "runner_up"),
        None,
        {"premium": premium, "primary": primary, "runner_up": runner_up},
        top, ladders, band, "deep", free_models,
    )


def auto_tiers(
    primary: str,
    premium: str,
    budget: str,
    runner_up: str,
    micro=(),
    top=None,
    ladders=None,
    band: float = SPREAD,
    specialized=None,
    free_models=(),
):
    micro = list(micro)[:2] + [""] * (2 - len(list(micro)[:2]))
    top = top or {}
    specialized = specialized or {}
    return _compact({
        T_SIMPLE: [
            (budget, "medium"), (micro[0], "low"), (micro[1], "low"), (primary, "low")],
        T_CHAT: [(budget, "high"), (primary, "medium"), (premium, "medium")],
        T_CODE: [(primary, "medium"), (runner_up, "medium"), (premium, "medium")],
        T_OPS: [(primary, "high"), (premium, "high"), (runner_up, "high")],
        FRONTIER: _frontier(
            premium, primary, runner_up, top, ladders, band, free_models
        ),
        T_VISION: [(primary, "medium"), (premium, "medium")],
        # Media generators answer no chat probe and declare no input window, so
        # they are listed by modality and carry no effort setting.
        T_IMAGE: [(name, None) for name in specialized.get("image", ())],
        T_VIDEO: [(name, None) for name in specialized.get("video", ())],
    })


def classify_tiers(primary: str, premium: str, budget: str, runner_up: str, micro=()):
    micro = list(micro)[:2] + [""] * (2 - len(list(micro)[:2]))
    return _compact({
        "SIMPLE": [(budget, "medium"), (micro[0], "low"), (micro[1], "low"), (primary, "low")],
        "MEDIUM": [(budget, "high"), (micro[0], "medium"), (micro[1], "medium"), (primary, "medium")],
        "COMPLEX": [(primary, "low"), (runner_up, "low"), (premium, "low")],
        "REASONING": [(primary, "medium"), (runner_up, "medium"), (premium, "medium")],
    })


def strix_tiers(primary: str, premium: str, budget: str, runner_up: str, micro=()):
    """Security work: no tier may be so cheap that it cannot read code.

    A scan that misses a hardcoded secret is worse than no scan, so even the light
    tier keeps a code-literate budget model, and exploit reproduction starts at
    the primary solver instead of a micro model.
    """
    micro = list(micro)[:1]
    return _compact({
        "SIMPLE": [(budget, "medium"), (micro[0] if micro else "", "low"), (primary, "low")],
        "MEDIUM": [(primary, "medium"), (runner_up, "medium"), (budget, "high")],
        "COMPLEX": [(primary, "high"), (premium, "high"), (runner_up, "high")],
        "REASONING": [(premium, "xhigh"), (primary, "high"), (runner_up, "high")],
    })


def advisor_tiers(primary: str, premium: str, budget: str, runner_up: str, micro=()):
    """Monitoring: cheapest model that answers, escalation only when a case is hard.

    Drift checks and regression triage arrive in volume, so the light tiers stay on
    the budget model; root-cause and guardrail judgement need the primary and
    premium solvers.
    """
    micro = list(micro)[:2] + [""] * (2 - len(list(micro)[:2]))
    return _compact({
        "SIMPLE": [(budget, "medium"), (micro[0], "low"), (micro[1], "low")],
        "MEDIUM": [(budget, "high"), (primary, "medium"), (runner_up, "low")],
        "COMPLEX": [(primary, "medium"), (runner_up, "medium"), (premium, "medium")],
        "REASONING": [(primary, "high"), (premium, "high")],
    })


def contract(tier_map: dict, definitions: dict, examples: dict, prompt: str, fallback: str) -> dict:
    """Classifier contract that must travel with the tier map.

    LiteLLM rejects a tier map that does not cover every defined tier, so the
    definitions, examples, prompt and fallback are derived from the *published*
    tiers and written in the same patch. The fallback is the catch-all, so a
    classifier that errors still lands on a tier that can answer.
    """
    published = [tier for tier in tier_map if tier in definitions]
    return {
        "tier_definitions": [
            {"name": tier, "description": definitions[tier]} for tier in published
        ],
        "classification_examples": "\n".join(
            examples[tier] for tier in published if tier in examples
        ),
        "classification_prompt": prompt,
        "fallback_tier": fallback if fallback in published else published[0],
    }


def policies(maps: dict) -> dict:
    """Per-router settings written in the same patch as the tiers.

    Only ``auto`` carries operator-defined tiers and its own classifier prompt:
    the engine accepts at most one such router, so the focus routers keep the
    shipped taxonomy and receive the routing policy alone.
    """
    out = {}
    if "auto" in maps:
        out["auto"] = contract(
            maps["auto"], AUTO_DEFINITIONS, AUTO_EXAMPLES, AUTO_PROMPT, T_CHAT
        )
    for name in ("auto-strix", "auto-advisor"):
        if name in maps:
            out[name] = {}
    for settings in out.values():
        settings.update(POLICY)
    return out


def routers(
    primary: str,
    premium: str,
    budget: str,
    runner_up: str,
    micro=(),
    top=None,
    ladders=None,
    band: float = SPREAD,
    per_tier: int | None = None,
    windows=None,
    specialized=None,
    free_models=(),
):
    """Map router name -> tier spec.

    ``top`` maps a model to its highest measured effort. ``ladders`` maps a model
    to its measured ladder so the top tier can be placed for equal score and every
    tier padded with same-score fallbacks; ``windows`` keeps that padding from
    shrinking a tier's declared window, and ``specialized`` supplies the
    image/video generators that only their own tiers may route to.
    """
    media = {name for names in (specialized or {}).values() for name in names}

    def allow(tier, name):
        # A text chain must never fall back to an image generator, and a media
        # tier must never fall back to a chat model.
        return name in media if tier in MEDIA_TIERS else name not in media

    maps = {
        "auto": auto_tiers(
            primary, premium, budget, runner_up, micro, top, ladders, band,
            specialized, free_models,
        ),
        "auto-classify": classify_tiers(primary, premium, budget, runner_up, micro),
        "auto-strix": strix_tiers(primary, premium, budget, runner_up, micro),
        "auto-advisor": advisor_tiers(primary, premium, budget, runner_up, micro),
    }
    profiles = {
        "auto": TIER_PROFILE,
        "auto-classify": CLASSIFY_PROFILE,
        "auto-strix": STRIX_PROFILE,
        "auto-advisor": ADVISOR_PROFILE,
    }
    if ladders:
        maps = {
            name: grow(
                tier_map, ladders, per_tier, band, windows, profiles[name], allow,
                # The classification group keeps its order: a free model is a
                # cost win for routed work, but the classifier needs stable,
                # predictable labels, not the cheapest model that answers.
                () if name == "auto-classify" else free_models,
            )
            for name, tier_map in maps.items()
        }
    return maps


def build(tier_map: dict) -> tuple[dict, dict]:
    """``(tiers, tier_model_configs)`` exactly as LiteLLM stores them."""
    tiers = {name: [model for model, _ in entries] for name, entries in tier_map.items()}
    configs = {
        name: [
            {
                "model_name": model,
                "litellm_params": (
                    {"reasoning_effort": effort} if effort else {}
                ),
            }
            for model, effort in entries
        ]
        for name, entries in tier_map.items()
    }
    return tiers, configs
