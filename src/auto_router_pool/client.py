"""HTTP access to LiteLLM and CLIProxyAPI.

Both speak the OpenAI-compatible API; only the base URL and bearer key differ.
HTTP errors are returned as ``{"error_status": code}`` instead of raising, so a
blocked or rate-limited model is a data point rather than a crash.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class Client:
    def __init__(self, base: str, key: str, timeout: float = 120):
        self.base = base.rstrip("/")
        self.key = key
        self.timeout = timeout

    def request(self, path: str, body=None, method=None):
        data = json.dumps(body).encode() if body is not None else None
        headers = {
            "Authorization": "Bearer " + self.key,
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(
            self.base + path, data=data, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            return {"error_status": error.code}

    def get(self, path: str):
        return self.request(path)

    def post(self, path: str, body):
        return self.request(path, body, "POST")

    def patch(self, path: str, body):
        return self.request(path, body, "PATCH")


def models(client: Client) -> list[dict]:
    payload = client.get("/model/info")
    return payload.get("data", [])


def probe(client: Client, model: str):
    """One-token round trip.

    Returns ``(ok, reason)``. 429/5xx and fallback-served answers are transient:
    a rate-limited model must not be demoted.
    """
    result = client.post(
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    status = result.get("error_status")
    if status in (400, 403, 404):
        return False, f"probe_{status}"
    return True, "ok" if status is None else f"transient_{status}"
