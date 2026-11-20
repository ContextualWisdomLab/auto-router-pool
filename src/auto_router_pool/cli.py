"""Command line entry point.

Nothing environment-specific is baked in: ports, model family, micro models and
the state directory all come from flags or environment variables, so this file
is safe to publish and reusable against any OpenAI-compatible proxy pair.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import aa
from .apply import apply as apply_changes
from .apply import (
    backup,
    classifier_drift,
    context_drift,
    plan,
    set_auto_fallback,
    smoke,
    validate,
)
from .client import Client, models, probe
from .pool import (
    MIN_INPUT_TOKENS,
    changed_scores,
    fingerprint,
    group_state,
    input_window,
    pick,
    score,
    score_hashes,
)
from .tiers import ROUTERS, routers


def default_home() -> Path:
    override = os.environ.get("AUTO_POOL_HOME")
    if override:
        return Path(override)
    return Path.home() / ".local" / "state" / "auto-router-pool"


@dataclass
class Settings:
    litellm_base: str = "http://127.0.0.1:4000"
    cliproxy_base: str = "http://127.0.0.1:8317"
    aa_base: str = aa.DEFAULT_BASE
    aa_ttl: float = aa.DEFAULT_TTL
    micro: tuple[str, ...] = ()
    home: Path = field(default_factory=default_home)
    key: str = ""
    cpkey: str = ""

    @classmethod
    def from_env(cls, env_file: Path | None = None):
        load_env(env_file)
        return cls(
            litellm_base=os.environ.get("LITELLM_BASE", cls.litellm_base),
            cliproxy_base=os.environ.get("CLIPROXY_BASE", cls.cliproxy_base),
            aa_base=os.environ.get("AA_BASE", cls.aa_base),
            aa_ttl=float(os.environ.get("AA_TTL", cls.aa_ttl)),
            key=os.environ.get("LITELLM_MASTER_KEY", ""),
            cpkey=os.environ.get("CPAPI_API_KEY", ""),
        )


def load_env(path: Path | None = None):
    """Read KEY=VALUE pairs without overwriting the real environment."""
    path = path or (os.environ.get("AUTO_POOL_ENV_FILE") or None)
    if not path:
        return
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"env file not found: {path}")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


def discover(client: Client, prefix: str = "", suffixes=()) -> list[str]:
    """Catalogue model ids, optionally narrowed by name prefix and suffix."""
    payload = client.get("/v1/models")
    names = [item["id"] for item in payload.get("data", [])]
    allowed = tuple(suffixes)
    found = []
    for name in names:
        if "/" in name:
            continue
        if prefix and not name.startswith(prefix):
            continue
        if allowed and not name.endswith(allowed):
            continue
        found.append(name)
    return sorted(set(found))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_state(path: Path, fingerprint_value: str, hashes: dict[str, str]):
    path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint_value,
                "scores": hashes,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        + "\n"
    )


def csv(value) -> tuple[str, ...]:
    return tuple(part.strip() for part in (value or "").split(",") if part.strip())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write changes (default: preview)")
    parser.add_argument("--watch", action="store_true", help="exit unless the catalogue changed")
    parser.add_argument("--refresh-scores", action="store_true", help="re-read every AA page")
    parser.add_argument("--smoke", action="store_true", help="one tiny auto completion after applying")
    parser.add_argument("--env-file", type=Path, default=None, help="file with KEY=VALUE secrets")
    parser.add_argument("--home", type=Path, default=None, help="state, cache and backup directory")
    parser.add_argument("--prefix", default="", help="only models whose id starts with this")
    parser.add_argument("--suffixes", default="", help="comma separated id suffixes, e.g. sol,astra")
    parser.add_argument("--micro", default=None, help="comma separated small models for light tiers")
    return parser.parse_args(argv)


def run(args) -> int:
    settings = Settings.from_env(args.env_file)
    if args.home:
        settings.home = args.home
    if args.micro is not None:
        settings.micro = csv(args.micro)
    missing = [
        name
        for name, value in (
            ("LITELLM_MASTER_KEY", settings.key),
            ("CPAPI_API_KEY", settings.cpkey),
        )
        if not value
    ]
    if missing:
        print(json.dumps({"error": "missing environment", "names": missing}))
        return 1

    settings.home.mkdir(parents=True, exist_ok=True)
    lock_path = settings.home / "auto-pool-refresh.lock"
    lock_path.touch()
    handle = lock_path.open("r")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(json.dumps({"skipped": "locked"}))
        return 0

    litellm = Client(settings.litellm_base, settings.key)
    cliproxy = Client(settings.cliproxy_base, settings.cpkey, timeout=30)

    rows = models(litellm)
    names = discover(cliproxy, args.prefix, csv(args.suffixes))
    if not names:
        print(json.dumps({"error": "no models discovered"}))
        return 1
    current_fp = fingerprint(rows, names)

    state_path = settings.home / "auto-pool-state.json"
    cache_path = settings.home / "auto-pool-aa-cache.json"
    state = load_json(state_path)
    if (
        args.watch
        and state.get("fingerprint") == current_fp
        and not context_drift(rows, ROUTERS)
        and not classifier_drift(rows, ROUTERS)
    ):
        print(json.dumps({"skipped": "no catalogue change", "fingerprint": current_fp[:12]}))
        return 0

    cache = load_json(cache_path)
    previous_scores = state.get("scores", {})

    status: dict[str, str] = {}
    candidates: dict[str, dict] = {}
    structural: list[str] = []
    for name in names:
        usable, reason = group_state(rows, name)
        if not usable:
            status[name] = reason
            continue
        ok, reason = probe(litellm, name)
        status[name] = reason
        if not ok:
            continue
        variants, error = aa.get_variants(
            name.replace(".", "-"),
            cache,
            base=settings.aa_base,
            ttl=settings.aa_ttl,
            force=args.refresh_scores,
        )
        if error in aa.STRUCTURAL and not variants:
            structural.append(name)
            status[name] = error
        elif not variants:
            status[name] = "no_aa_entry"
        else:
            candidates[name] = variants

    if structural:
        # Deliberately no state write: the watcher keeps retrying instead of
        # pinning a fingerprint that would hide the broken model forever.
        print(
            json.dumps(
                {"result": "aa_unparseable", "models": structural, "status": status},
                ensure_ascii=False,
            )
        )
        return 1

    hashes = score_hashes(cache)
    updates = changed_scores(hashes, previous_scores)
    cache_path.write_text(json.dumps(cache))

    premium = pick(candidates, "xhigh", 12, 45)
    primary = pick(candidates, "high", 3, 12)
    budget = pick(candidates, "medium", 0.30, 1.50, mode="value")
    if not (premium and primary and budget):
        print(json.dumps({"error": "insufficient valid candidates", "status": status}))
        return 1

    eligible = sorted(
        (
            (name, score(variants, "high"))
            for name, variants in candidates.items()
            if score(variants, "high")
            and score(variants, "high")[2]["in"] <= 3
            and score(variants, "high")[2]["out"] <= 12
        ),
        key=lambda item: item[1][0],
        reverse=True,
    )
    runner_up = next((name for name, _ in eligible if name != primary[0]), None)
    if not runner_up:
        print(json.dumps({"error": "no runner-up candidate", "status": status}))
        return 1

    # A micro model below the context floor cannot serve this router at all, so
    # it is dropped from the light tiers instead of loweriing their window.
    micro = [name for name in settings.micro if input_window(rows, name) >= MIN_INPUT_TOKENS]
    for name in settings.micro:
        if name not in micro:
            status[name] = "context_below_floor"
    changes = plan(
        rows,
        routers(primary[0], premium[0], budget[0], runner_up, micro),
        primary[0],
        runner_up,
    )
    summary = {
        "pool": {
            "premium": premium[0],
            "primary": primary[0],
            "budget": budget[0],
            "runner_up": runner_up,
        },
        "index": {
            "premium": round(premium[2], 1),
            "primary": round(primary[2], 1),
            "budget": round(budget[2], 1),
        },
        "score_updates": updates,
        "status": status,
        "minimum_input_tokens": MIN_INPUT_TOKENS,
        "changed": len(changes),
        "max_input_tokens": {
            change["model_name"]: change["model_info"]["max_input_tokens"]
            for change in changes
            if change.get("model_info")
        },
    }

    if not changes:
        save_state(state_path, current_fp, hashes)
        print(json.dumps({**summary, "result": "unchanged"}, ensure_ascii=False))
        return 0

    try:
        validate(litellm, changes)
        backup_path = backup(changes, settings.home)
        if not args.apply:
            print(
                json.dumps(
                    {**summary, "result": "preview", "backup": str(backup_path)},
                    ensure_ascii=False,
                )
            )
            return 0

        apply_changes(litellm, changes)
        set_auto_fallback(litellm, [premium[0], primary[0], runner_up, budget[0]])
    except (ValueError, RuntimeError) as error:
        print(json.dumps({**summary, "result": "rejected", "error": str(error)}))
        return 1

    save_state(state_path, current_fp, hashes)
    output = {**summary, "result": "applied", "backup": str(backup_path)}
    if args.smoke:
        output["smoke"] = smoke(litellm)
    print(json.dumps(output, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
