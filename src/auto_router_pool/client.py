"""HTTP access to LiteLLM and CLIProxyAPI.

Both speak the OpenAI-compatible API; only the base URL and bearer key differ.
HTTP errors are returned as ``{"error_status": code}`` instead of raising, so a
blocked or rate-limited model is a data point rather than a crash.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

# One token should not take a full admin timeout; a hung deployment must not
# stall the 15-minute refresh. Long enough for a slow reasoning first token.
PROBE_TIMEOUT = 30


class Client:
    def __init__(self, base: str, key: str, timeout: float = 120):
        self.base = base.rstrip("/")
        self.key = key
        self.timeout = timeout

    def request(self, path: str, body=None, method=None, timeout: float | None = None):
        data = json.dumps(body).encode() if body is not None else None
        headers = {
            "Authorization": "Bearer " + self.key,
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(
            self.base + path, data=data, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            try:
                body = json.loads(error.read())
            except Exception:
                body = None
            return {"error_status": error.code, "error_body": body}
        except (urllib.error.URLError, TimeoutError, OSError):
            # A hung deployment must be a data point, not a crashed run.
            return {"error_status": "unreachable"}

    def get(self, path: str, timeout: float | None = None):
        return self.request(path, timeout=timeout)

    def post(self, path: str, body, timeout: float | None = None):
        return self.request(path, body, "POST", timeout)

    def patch(self, path: str, body):
        return self.request(path, body, "PATCH")


BUDGET_WORDS = (
    "budget has been exceeded",
    "budget exceeded",
    "budget_exceeded",
    "exceededbudget",
)


def budget_exceeded(error) -> bool:
    """Whether an upstream answer is LiteLLM's budget-exceeded error.

    Budget is enforced per key, so this is the one failure that says every
    deployment sharing the credential is unusable, not just the one that
    answered. The check is on the message, never on a credential name.
    """
    text = str(error or "").lower()
    return any(word in text for word in BUDGET_WORDS) or (
        "budget" in text and "exceed" in text
    )


def deployment_health_detail(
    client: Client, model_id: str, timeout: float = 25
) -> tuple[str, str]:
    """One health sample plus the lower-cased upstream error snippet, if any."""
    result = client.get(
        "/health?" + urllib.parse.urlencode({"model_id": model_id}), timeout=timeout
    )
    body = result.get("error_body")
    if not isinstance(body, dict):
        body = result

    def reported(key: str) -> list[dict]:
        endpoints = body.get(key) or []
        # ``?model_id=`` already scopes the report, and LiteLLM omits the id
        # from each endpoint. Only filter when ids are actually present.
        scoped = [
            endpoint
            for endpoint in endpoints
            if endpoint.get("model_id") == model_id
        ]
        if scoped:
            return scoped
        if any(endpoint.get("model_id") for endpoint in endpoints):
            return []
        return endpoints

    if reported("healthy_endpoints"):
        return "ok", ""
    unhealthy = reported("unhealthy_endpoints")
    if unhealthy:
        error = str(unhealthy[0].get("error", "")).lower()
        if budget_exceeded(error):
            return "budget_exceeded", error
        if any(
            marker in error
            for marker in ("429", "rate limit", "rate_limit", "usage limit", "cooling down")
        ):
            return "rate_limited", error
        return "unhealthy", error
    if result.get("error_status") == 429:
        return "rate_limited", ""
    return "unknown", ""


def deployment_health(client: Client, model_id: str, timeout: float = 25) -> str:
    """One health sample for a single deployment.

    ``ok`` means the endpoint answered; ``rate_limited`` means it was a 429 or
    reported as cooling down. The health route may return 429 with an endpoint
    report in the error body, so the body is read before the HTTP status.
    """
    return deployment_health_detail(client, model_id, timeout=timeout)[0]


DECISION_PATHS = ("/v1/decisions", "/v1/systemone")


# The smallest well-formed choice question: one label has to come back, which is
# the same shape the router asks the classifier for a tier.
# The wire shape LiteLLM validates: a choice question carries option objects, each
# with a ``value``, and nothing nests under a ``choice`` key.
DECISION_CHOICE = {
    "questions": [
        {
            "type": "choice",
            "instructions": "Which routing tier fits this request?",
            "choices": [
                {"value": "SIMPLE", "description": "a single lookup or rewrite"},
                {"value": "MEDIUM", "description": "an ordinary implementation request"},
                {"value": "COMPLEX", "description": "multi-step work across several files"},
            ],
        }
    ]
}


# Cloudflare Clef answers the native System One route only, and says so when asked
# on the OpenAI-shaped one, so both have to be tried before a model is written off.
DECISION_SYSTEMONE = {
    "state": "write a for loop",
    "questions": {
        "tier": {
            "type": "choice",
            "instructions": "Which routing tier fits this request?",
            "criteria": {
                "SIMPLE": "a single lookup or rewrite",
                "MEDIUM": "an ordinary implementation request",
                "COMPLEX": "multi-step work across several files",
            },
        }
    },
}


def probe_decision_choice(client: Client, model: str):
    """Whether a deployment answers a real choice question.

    Two routes exist for the same capability: the OpenAI-shaped ``/v1/decisions``
    and the native ``/v1/systemone``. A registration can claim either, and a
    provider can accept the route while refusing the shape, so only a 200 proves
    the classifier will work -- anything else keeps the LLM judge, because a
    classifier that cannot answer would break routing outright.
    """
    last = "no_decision_route"
    for path, extra in (
        ("/v1/decisions", {"input": "write a for loop", **DECISION_CHOICE}),
        ("/v1/systemone", DECISION_SYSTEMONE),
    ):
        result = client.post(
            path, {"model": model, **extra}, timeout=PROBE_TIMEOUT
        )
        status = result.get("error_status")
        if status is None:
            return True, f"ok_{path.rsplit('/', 1)[-1]}"
        last = f"unavailable_{status}"
        if status not in (400, 401, 403, 404):
            # A timeout or 5xx will not be fixed by the other route.
            break
    return False, last


def probe_decision(client: Client, model: str):
    """Whether a decision endpoint serves this model.

    A decision model answers a schema, not a chat turn, and the schema differs
    per vendor, so a shape error (400/422) still proves the route exists -- that
    is exactly the proof a chat probe cannot give. An auth or alias error means
    the registration itself is wrong and must not be kept.
    """
    last = "no_decision_route"
    for path in DECISION_PATHS:
        result = client.post(
            path, {"model": model}, timeout=PROBE_TIMEOUT
        )
        status = result.get("error_status")
        if status is None:
            return True, "ok"
        if status in (400, 422):
            return True, f"schema_{status}"
        last = f"unavailable_{status}"
        if status in (401, 403, 404):
            continue
        return False, last
    return False, last


def models(client: Client) -> list[dict]:
    payload = client.get("/model/info")
    return payload.get("data", [])


def _number(value) -> float | None:
    """A numeric usage field, tolerating providers that JSON-encode it as text."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def usage_free(usage: dict | None) -> bool | None:
    """Whether one completed call was billed as free, from its ``usage`` object.

    LiteLLM returns ``usage.cost`` and ``cost_details`` for a zero-priced free
    tier; OpenAI paid answers sometimes omit the cost fields entirely, while
    OpenRouter and Anthropic include them. Only positive evidence is used: a
    missing field is ``None`` (unknown), not paid. BYOK is not a free tier even
    when the proxy's own cost is zero.
    """
    if not isinstance(usage, dict):
        return None
    if usage.get("is_byok") is True:
        return False
    details = usage.get("cost_details")
    numbers = []
    if isinstance(details, dict):
        numbers = [number for value in details.values() if (number := _number(value)) is not None]
    cost = _number(usage.get("cost"))
    if cost is not None:
        if cost > 0:
            return False
        if numbers and any(value > 0 for value in numbers):
            return False
        return True
    if numbers:
        return all(value == 0 for value in numbers)
    return None


def probe_with_usage(client: Client, model: str):
    """``probe`` plus the free/paid signal from the probe's ``usage`` object.

    The one-token probe is already the cheapest point at which the gateway
    knows which deployment answered, so its usage is the right place to learn
    whether a free tier was actually applied. ``observation`` is ``None`` when
    the response carried no cost information.
    """
    result = client.post(
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        },
        timeout=PROBE_TIMEOUT,
    )
    status = result.get("error_status")
    if status is not None:
        return False, f"unavailable_{status}", None
    served = result.get("model")
    if served and served != model:
        # A fallback answer proves only that the fallback is alive.
        return False, f"fallback_{served}", None
    usage = result.get("usage")
    observation = None
    if isinstance(usage, dict):
        observation = {
            "free": usage_free(usage),
            "cost": usage.get("cost"),
        }
    return True, "ok", observation


def probe(client: Client, model: str):
    """One-token round trip that must be served by ``model`` itself.

    Returns ``(ok, reason)``. Any error (429/5xx/timeout), or a 200 answered by a
    fallback, means the model cannot serve this run, so it drops out of the
    candidates. Nothing is written to the catalogue, so the next run re-admits
    the model as soon as it answers again.
    """
    ok, reason, _ = probe_with_usage(client, model)
    return ok, reason
