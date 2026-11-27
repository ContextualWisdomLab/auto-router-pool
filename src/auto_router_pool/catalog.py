"""Register upstream catalogue models the gateway does not carry yet.

A model missing from the registry is invisible to the router however good it is,
and the upstream catalogue is the only place a model exists before anyone adds
it. This stage plans one deployment per eligible catalogue model and writes only
registrations: block flags stay with the quarantine sweep, because a probe can
prove a deployment is dead but cannot prove an account is out of credit.

The transport is chosen from what the catalogue says the model *is*. A model
that serves decisions natively (``/v1/systemone``, "decision model") cannot use
the OpenAI chat transport: that sends ``/v1/decisions`` and the provider refuses
it. Such a model gets the generic SystemOne transport instead, which keeps the
model id and posts the native ``state``/``questions`` body.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.experientiallabs.ai"
CREDENTIAL = "experientiallabs-shared"
OWNER = "auto_pool_catalog"
# The block-flipping predecessor owned the same rows; adopting them migrates the
# transport (clef) instead of leaving a second, dead deployment behind.
LEGACY_OWNER = "experientiallabs-sync-v1"
# The registry was 396 rows when this stage was added; leave headroom for the
# full catalogue instead of failing the whole run at the old 512/768 limits.
MAX_DEPLOYMENTS = 1024

DECISION_WORDS = ("decision model", "systemone")
COMPATIBLE = {"openai", "xai", "moonshot", "deepseek", "zai", "cognition"}
PUBLISHERS = {
    "openai": ("OpenAI", "openai"), "anthropic": ("Anthropic", "anthropic"),
    "google": ("Google", "gemini"), "gemini": ("Google", "gemini"),
    "xai": ("xAI", "xai"), "x.ai": ("xAI", "xai"), "x-ai": ("xAI", "xai"),
    "moonshot": ("Moonshot AI", "moonshot"), "moonshot ai": ("Moonshot AI", "moonshot"),
    "moonshotai": ("Moonshot AI", "moonshot"),
    "deepseek": ("DeepSeek", "deepseek"),
    "z.ai": ("Z.ai", "zai"), "z-ai": ("Z.ai", "zai"), "zai": ("Z.ai", "zai"),
    "cognition": ("Cognition", "cognition"),
}

# Gemini, Anthropic, Z.ai and xAI refuse a request that carries
# prompt_cache_key; OpenAI still accepts it.
DROP_PARAM = "prompt_cache_key"
DROP_PROVIDERS = frozenset({"gemini", "anthropic", "zai", "xai"})
DROP_NAME_TOKENS = ("gemini", "claude", "glm", "grok")


def load_key(home: Path, environ=None) -> str | None:
    """The Experiential Labs key, from the environment or the sync env file."""
    environ = os.environ if environ is None else environ
    value = environ.get("EXPERIENTIALLABS_API_KEY")
    if value:
        return value
    path = Path(home) / "experientiallabs-sync.env"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        name, separator, value = line.strip().removeprefix("export ").partition("=")
        if separator and name.strip().lower() == "experientiallabs_api_key":
            value = value.strip().strip("\"'")
            if value:
                return value
    return None


def _get_json(url: str, key: str | None = None, timeout: float = 60):
    headers = {"Authorization": "Bearer " + key} if key else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def fetch(api: str = API, key: str = "", timeout: float = 60) -> tuple[set[str], dict]:
    """The models this account can call, and the public catalogue rows.

    The authenticated list is the authority for what can be called at all: the
    public catalogue also advertises models the account cannot reach, and
    registering one of those creates a group that can never answer.
    """
    auth = _get_json(api + "/v1/models", key, timeout)
    ids = {item.get("id") for item in auth.get("data") or [] if item.get("id")}
    public: dict[str, dict] = {}
    offset = 0
    while True:
        page = _get_json(
            api + "/api/models?" + urllib.parse.urlencode({"limit": 500, "offset": offset}),
            timeout=timeout,
        )
        models = page.get("models")
        if not isinstance(models, list):
            raise ValueError("invalid public catalogue page")
        for item in models:
            slug = ((item or {}).get("model") or {}).get("slug")
            if slug:
                public[slug] = item
        offset += len(models)
        if offset >= page.get("total", 0) or not models:
            break
        if offset > 20_000:
            raise ValueError("public catalogue did not terminate")
    if not ids or not public:
        raise ValueError("catalogue is empty")
    return ids, public


def route(entry: dict) -> dict | None:
    """The first routable provider route for a model, if any."""
    routes = [
        item for item in (entry.get("providers") or [])
        if isinstance(item, dict) and item.get("routable")
    ]
    return routes[0] if routes else None


def decisions_endpoint(entry: dict) -> str | None:
    """Which decision API the catalogue says this model speaks, if any.

    Two shapes exist. A native model takes ``state`` + ``questions`` on
    ``/v1/systemone``; an OpenAI-shaped model takes ``input`` + ``questions`` on
    ``/v1/decisions``. They need different transports, and a model that also
    serves chat is capable of deciding but is not decision-only.
    """
    model = entry.get("model") or {}
    endpoints = [str(item).lower() for item in (model.get("supported_endpoints") or [])]
    text = " ".join(
        str(model.get(field) or "") for field in ("description", "display_name", "availability")
    ).lower()
    if any("systemone" in item for item in endpoints) or "systemone" in text:
        return "/v1/systemone"
    decisions = any("/decisions" in item for item in endpoints)
    chat = any("chat/completions" in item for item in endpoints)
    if decisions and not chat:
        return "/v1/decisions"
    # A vendor may state exclusivity in prose instead of an endpoint list.
    if "/decisions" in text and any(
        marker in text for marker in ("decision-only", "not a chat")
    ):
        return "/v1/decisions"
    return None


def is_decisions_model(entry: dict) -> bool:
    """Whether the catalogue describes a decision-only model.

    The endpoint matters for the transport: registering a native model through
    the OpenAI chat transport would send ``/v1/decisions`` and be refused, while
    an OpenAI-shaped decision model must keep that route.
    """
    model = entry.get("model") or {}
    if str(model.get("mode") or "").lower() in {"decision", "decisions", "evaluation"}:
        return True
    return decisions_endpoint(entry) is not None


def prompt_cache_drop(params: dict) -> dict:
    """Add prompt_cache_key to a deployment that would be sent it and refuse it."""
    model = str(params.get("model") or "")
    provider = model.split("/", 1)[0] if "/" in model else ""
    tail = model.split("/")[-1].lower()
    if provider not in DROP_PROVIDERS and not any(t in tail for t in DROP_NAME_TOKENS):
        return params
    drop = [item for item in (params.get("additional_drop_params") or []) if isinstance(item, str)]
    if DROP_PARAM in drop:
        return params
    return {**params, "additional_drop_params": drop + [DROP_PARAM]}


def resolve_publisher(slug: str, model: dict, provider_route: dict, prices: dict | None):
    """``(publisher, transport)`` from catalogue metadata.

    Only transports verified against the upstream API are named; anything else
    goes through the OpenAI-compatible surface, which the Experiential host
    exposes for every chat model in its catalogue.
    """
    maker = model.get("maker")
    maker = maker.strip() if isinstance(maker, str) else ""
    if maker.casefold() in {"", "null", "none", "unknown"}:
        maker = ""
    publisher, native = PUBLISHERS.get(maker.casefold(), (maker, None))
    provider_id = provider_route.get("provider_model_id")
    provider_id = provider_id.lstrip("~") if isinstance(provider_id, str) else ""
    prefix, separator, tail = provider_id.partition("/")
    if not maker and separator:
        publisher, native = PUBLISHERS.get(prefix.casefold(), (prefix, None))
    if not publisher and not native and isinstance(prices, dict):
        canonical = next(
            (
                key for key in (slug, provider_id, tail)
                if (prices.get(key) or {}).get("mode") == "chat"
            ),
            None,
        )
        if canonical:
            native = (prices.get(canonical) or {}).get("litellm_provider")
            publisher, native = PUBLISHERS.get(str(native).casefold(), ("", None))
    transport = native if native in COMPATIBLE or native == "anthropic" else "openai"
    return publisher or "Unknown", transport


def _same(previous: dict | None, body: dict) -> bool:
    if not previous:
        return False
    params = previous.get("litellm_params") or {}
    info = previous.get("model_info") or {}
    return (
        all(params.get(key) == value for key, value in body["litellm_params"].items())
        and all(
            info.get(key) == value
            for key, value in body["model_info"].items()
            if key != "id"
        )
    )


def plan(auth_ids, public: dict, rows: list[dict], prices: dict | None = None,
         retired=()) -> list[tuple[str, dict]]:
    """One planned action per eligible catalogue model.

    ``create`` and ``update`` bodies are ready to POST/PATCH; ``unchanged`` ones
    are reported but never written. Existing deployments owned by the retired
    sync are adopted so their transport can be corrected in place.
    """
    retired = set(retired)
    planned: list[tuple[str, dict]] = []
    creates = 0
    for slug in sorted(auth_ids):
        if slug in retired:
            continue
        entry = public.get(slug)
        # A free tier is the same model as its base and is often omitted from
        # the public catalogue. The authenticated list proves it can be called,
        # and the base entry supplies the metadata, so only derive that case.
        free = slug.endswith(":free")
        if entry is None:
            if not free or slug.removesuffix(":free") not in public:
                continue
            entry = public[slug.removesuffix(":free")]
        model = entry.get("model") or {}
        provider_route = route(entry)
        if not provider_route:
            continue
        if model.get("status") != "active":
            continue
        if "text" not in (model.get("output_modalities") or []):
            continue

        same = [row for row in rows if row.get("model_name") == slug]
        owned = [
            row for row in same
            if (row.get("model_info") or {}).get(OWNER) == OWNER
        ]
        legacy = [
            row for row in same
            if (row.get("model_info") or {}).get("experientiallabs_sync") == LEGACY_OWNER
            and (row.get("litellm_params") or {}).get("litellm_credential_name") == CREDENTIAL
        ]
        if len(owned) > 1 or len(legacy) > 1:
            raise ValueError("ambiguous owned route for " + slug)
        previous = (owned or legacy or [None])[0]
        if previous is None:
            order = max(
                (
                    row.get("litellm_params", {}).get("order", 1)
                    for row in same
                ),
                default=0,
            ) + 1
        else:
            order = previous.get("litellm_params", {}).get("order", 1)

        publisher, transport = resolve_publisher(slug, model, provider_route, prices)
        endpoint = decisions_endpoint(entry)
        if endpoint == "/v1/systemone":
            params = {
                "model": "hosted_vllm/" + slug,
                "custom_llm_provider": "hosted_vllm",
            }
        elif endpoint == "/v1/decisions":
            # OpenAI's decisions shape is served by the OpenAI transport; only
            # the native SystemOne shape needs the generic transport.
            params = {
                "model": "openai/" + slug,
                "custom_llm_provider": "openai",
            }
        else:
            params = {
                "model": transport + "/" + slug,
                "custom_llm_provider": transport,
            }
        endpoints = [endpoint] if endpoint else None
        params.update({
            "api_base": API if transport == "anthropic" else API + "/v1",
            "litellm_credential_name": CREDENTIAL,
            "order": order,
        })
        in_nano = provider_route.get("input_nano_usd_per_million")
        out_nano = provider_route.get("output_nano_usd_per_million")
        if free:
            params["input_cost_per_token"] = 0.0
            params["output_cost_per_token"] = 0.0
        elif isinstance(in_nano, int) and isinstance(out_nano, int):
            params["input_cost_per_token"] = in_nano / 1e15
            params["output_cost_per_token"] = out_nano / 1e15
        params = prompt_cache_drop(params)

        info = {
            "mode": "decisions" if endpoint else "chat",
            **(({"is_free": True}) if free else {}),
            OWNER: OWNER,
            "catalog_slug": slug,
            "original_maker": publisher,
            "description": (
                str(model.get("description") or model.get("display_name") or slug)
                + " (Publisher: " + publisher + ")"
            ),
            "availability": "authenticated Experiential Labs catalog",
        }
        if (
            free
            and isinstance(in_nano, int) and not isinstance(in_nano, bool)
            and isinstance(out_nano, int) and not isinstance(out_nano, bool)
        ):
            # A free tier is the same model as its base; keep the list price so
            # priority can compare it without treating zero as a real price.
            info["original_input_cost_per_token"] = in_nano / 1e15
            info["original_output_cost_per_token"] = out_nano / 1e15
        if endpoints:
            info["supported_endpoints"] = endpoints
        for field in ("max_input_tokens", "max_output_tokens"):
            value = model.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                info[field] = value
        if previous:
            info["id"] = previous.get("model_info", {}).get("id")
        body = {"model_name": slug, "litellm_params": params, "model_info": info}
        if previous and info.get("id"):
            action = "unchanged" if _same(previous, body) else "update"
        else:
            action = "create"
            creates += 1
        planned.append((action, body))

    if len(rows) + creates > MAX_DEPLOYMENTS:
        raise ValueError("registry ceiling would be exceeded")
    return planned


def apply_plan(client, planned) -> list[dict]:
    """Write created and updated deployments; report every attempted action."""
    results = []
    for action, body in planned:
        name = body["model_name"]
        if action == "create":
            result = client.post("/model/new", body)
            status = result.get("error_status")
            record = {"model": name, "action": "created" if status is None else "failed"}
        elif action == "update":
            identifier = body["model_info"]["id"]
            result = client.patch(f"/model/{urllib.parse.quote(identifier, safe='')}/update", body)
            status = result.get("error_status")
            record = {"model": name, "action": "updated" if status is None else "failed"}
        else:
            continue
        if status is not None:
            record["detail"] = f"error_{status}"
        results.append(record)
    return results


def sync(client, rows: list[dict], home: Path, *, prices: dict | None = None,
         key: str | None = None, api: str = API, apply: bool = False) -> dict:
    """Fetch, plan and (when asked) register missing catalogue models.

    A missing key or an unreachable catalogue is reported, not raised: the
    routing run has to continue with the deployments it already has.
    """
    key = key or load_key(home)
    if not key:
        return {"source": "experientiallabs", "skipped": "no_credential"}
    try:
        auth_ids, public = fetch(api, key)
        retired_path = Path(home) / "retired-commercial-models.json"
        retired = json.loads(retired_path.read_text()) if retired_path.exists() else []
        if not isinstance(retired, list):
            raise ValueError("retired model list is invalid")
        planned = plan(auth_ids, public, rows, prices, retired)
    except (OSError, ValueError, urllib.error.URLError) as error:
        return {"source": "experientiallabs", "error": str(error)[:200]}

    changes = [(action, body) for action, body in planned if action != "unchanged"]
    summary = {
        "source": "experientiallabs",
        "models": len(auth_ids),
        "eligible": len(planned),
        "create": sum(action == "create" for action, _ in planned),
        "update": sum(action == "update" for action, _ in planned),
        "unchanged": sum(action == "unchanged" for action, _ in planned),
    }
    if apply:
        summary["actions"] = apply_plan(client, changes)
    else:
        summary["actions"] = [
            {"model": body["model_name"], "action": action}
            for action, body in changes
        ]
    return summary
