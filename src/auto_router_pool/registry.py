"""Guest models: ones the operator can use that the catalogue may not carry.

Some deployments are never discovered by a catalogue sync -- a free upstream
router, a decision endpoint, a preview alias. They are declared in
``auto-pool-guests.json`` and registered on demand, because a model the router
does not know about cannot be routed to at all.

Registration is not fire-and-forget: a guest that registers but cannot answer a
request is deleted again, so a stale declaration never leaves a dead model group
behind for the next run to probe.
"""
from __future__ import annotations

import json
from pathlib import Path

from .client import probe as chat_probe
from .client import probe_decision
from .pool import DECISION_MODES, is_batch

REQUIRED = ("model_name", "litellm_params")


def load_guests(path: Path) -> list[dict]:
    """Declared guests, validated before anything is written.

    A malformed declaration is an error, not a skipped entry: silently dropping
    it would leave the operator believing a model was registered.
    """
    if not path.exists():
        return []
    payload = json.loads(path.read_text() or "[]")
    guests = payload.get("models", []) if isinstance(payload, dict) else payload
    out = []
    for guest in guests:
        if not isinstance(guest, dict):
            raise ValueError(f"guest entry must be an object: {guest!r}")
        for key in REQUIRED:
            if not guest.get(key):
                raise ValueError(f"guest entry needs {key}: {guest!r}")
        if not (guest["litellm_params"] or {}).get("model"):
            raise ValueError(f"guest entry needs litellm_params.model: {guest!r}")
        out.append(guest)
    return out


def registered(rows: list[dict], name: str) -> bool:
    """Whether the catalogue already carries this model group."""
    return any(row["model_name"] == name for row in rows)


def ensure_guests(client, guests: list[dict], rows: list[dict], probe=chat_probe,
                  decision_probe=probe_decision):
    """Register every missing guest, keeping only the ones that answer.

    Returns one record per guest: ``present`` when the catalogue already had it,
    ``registered`` when it was created and answered, ``reverted`` when it was
    created but could not serve a request (and was deleted again), or ``failed``
    when the server refused the registration itself.
    """
    results = []
    for guest in guests:
        name = guest["model_name"]
        if registered(rows, name):
            results.append({"model": name, "action": "present"})
            continue

        created = client.post("/model/new", guest)
        status = created.get("error_status")
        if status is not None:
            results.append(
                {"model": name, "action": "failed", "detail": f"create_{status}"}
            )
            continue

        # A decision model is registered by its endpoint, whatever it is called:
        # the name proves nothing, so the route itself is what gets probed.
        mode = str((guest.get("model_info") or {}).get("mode") or "chat").lower()
        if mode in DECISION_MODES:
            ok, reason = decision_probe(client, name)
        elif is_batch(guest):
            results.append(
                {"model": name, "action": "registered", "detail": "no batch probe"}
            )
            continue
        elif mode != "chat":
            results.append(
                {"model": name, "action": "registered", "detail": f"no {mode} probe"}
            )
            continue
        else:
            ok, reason = probe(client, name)
        if ok:
            results.append({"model": name, "action": "registered"})
            continue

        client.post("/model/delete", {"id": created.get("model_id")})
        results.append({"model": name, "action": "reverted", "detail": reason})
    return results
