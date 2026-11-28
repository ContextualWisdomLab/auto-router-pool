import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from auto_router_pool import extra_catalogs as extra

PRICE = {
    "litellm_provider": "xai", "mode": "chat",
    "input_cost_per_token": 2e-6, "output_cost_per_token": 6e-6,
    "max_input_tokens": 500000, "max_output_tokens": 500000,
}
PRICES = {"xai/grok-4.6": PRICE, "grok-4-6": {**PRICE, "mode": "embedding"}}
CATALOG = {"data": [
    {"id": name, "owned_by": "cursor"} for name in (
        "cursor-grok-4.6", "cursor-grok-4.6-high", "cursor-grok-4.6-fast",
        "cursor-composer-2.5", "grok4.6",
    )
]}


class FakeClient:
    def __init__(self, info=None):
        self.calls = []
        self.info = info or []

    def get(self, path, timeout=None):
        self.calls.append(("GET", path))
        if path == "/model/info":
            return {"data": self.info}
        return {"items": [{"id": "cloud-a"}]}

    def post(self, path, body, timeout=None):
        self.calls.append(("POST", path, body))
        return {"model_id": "new-id"}

    def patch(self, path, body):
        self.calls.append(("PATCH", path, body))
        return {}


class CursorMatchTest(unittest.TestCase):
    def test_aliases_and_unpriced_models_are_audited_not_registered(self):
        approved, audit = extra.cursor_matches(CATALOG, PRICES)
        self.assertEqual([match["public"] for match in approved], ["grok-4.6"])
        self.assertEqual(len(audit), 4)
        body = extra.cursor_payload(approved[0], PRICES)
        self.assertEqual(body["model_name"], "grok-4.6")
        self.assertEqual(body["litellm_params"]["model"], "xai/cursor-grok-4.6")
        self.assertEqual(body["litellm_params"]["api_base"], extra.CURSOR_BRIDGE + "/v1")
        self.assertEqual(body["model_info"]["cursor_chat_sync"], extra.CURSOR_OWNER)
        self.assertEqual(body["model_info"]["max_input_tokens"], 500000)

    def test_a_retired_name_is_rejected(self):
        approved, audit = extra.cursor_matches(
            {"data": [{"id": "cursor-grok-4.6"}]}, PRICES, retired={"grok-4.6"}
        )
        self.assertEqual(approved, [])
        self.assertEqual(audit[0]["reason"], "retired non-current commercial model")

    def test_a_zero_price_is_not_a_verified_rate(self):
        approved, _ = extra.cursor_matches(
            {"data": [{"id": "cursor-grok-4.6"}]},
            {"xai/grok-4.6": {**PRICE, "input_cost_per_token": 0}},
        )
        self.assertEqual(approved, [])

    def test_family_strips_only_known_variant_suffixes(self):
        self.assertEqual(
            extra.cursor_family("cursor-grok-4.6-thinking-extra-high-fast"),
            ("grok-4.6", True),
        )
        self.assertEqual(extra.cursor_family("gpt-5.5-high-low")[0], "gpt-5.5-high")


class CursorReconcileTest(unittest.TestCase):
    def setUp(self):
        self.approved, _ = extra.cursor_matches(CATALOG, PRICES)
        self.match = self.approved[0]
        self.ident = extra.cursor_identity(self.match)

    def api(self, calls, registry, state):
        def call(path, body=None, method=None):
            calls.append((path, method))
            if path == "/model/new":
                self.assertTrue(state["deployments"][self.ident].get("pending"))
                row = json.loads(json.dumps(body))
                row["model_info"]["id"] = "server-uuid"
                registry.append(row)
                return row
            registry[0] = json.loads(json.dumps(body))
            return registry[0]
        return call

    def test_create_then_unchanged(self):
        registry, calls, saves = [], [], []
        state = {"owner": extra.CURSOR_OWNER, "deployments": {}}
        api = self.api(calls, registry, state)
        result = extra.cursor_reconcile(
            self.approved, PRICES, registry, state, api, saves.append
        )
        self.assertEqual(result["created"], 1)
        self.assertEqual(state["deployments"][self.ident]["id"], "server-uuid")
        again = extra.cursor_reconcile(
            self.approved, PRICES, registry, state, api, saves.append
        )
        self.assertEqual(again["unchanged"], 1)
        self.assertEqual(len(calls), 1)

    def test_a_manual_route_is_never_adopted(self):
        body = extra.cursor_payload(self.match, PRICES)
        row = json.loads(json.dumps(body))
        row["model_info"]["id"] = "manual"
        row["model_info"].pop("cursor_chat_sync")
        result = extra.cursor_reconcile(
            self.approved, PRICES, [row], {"deployments": {}}, lambda *a: {}, lambda *a: None
        )
        self.assertEqual(result["manual_preserved"], 1)


class ArgonTest(unittest.TestCase):
    def test_the_client_key_is_read_without_a_yaml_dependency(self):
        yaml_text = "port: 8317\napi-keys:\n  - 'secret-key'\n  - other\n"
        self.assertEqual(extra.config_api_keys(yaml_text), ["secret-key", "other"])
        json_text = json.dumps({"port": 8317, "api-keys": ["secret-key"]})
        self.assertEqual(extra.config_api_keys(json_text), ["secret-key"])
        self.assertEqual(extra.config_api_keys("api-keys: []\n"), [])

    def test_the_preferred_upstream_is_chosen(self):
        self.assertEqual(
            extra.choose_argon(["other", "gemini-4-argon-high"]),
            "gemini-4-argon-high",
        )
        self.assertIsNone(extra.choose_argon(["other"]))

    def test_plan_creates_with_sibling_metadata(self):
        rows = [{"model_name": "gemini-4-argon", "litellm_params": {"order": 2},
                 "model_info": {"supports_reasoning": True, "id": "sibling"}}]
        action, body = extra.plan_argon(rows, "gemini-4-argon-high")
        self.assertEqual(action, "create")
        self.assertEqual(body["litellm_params"]["order"], 1)
        self.assertEqual(body["litellm_params"]["model"], "gemini/gemini-4-argon-high")
        self.assertTrue(body["model_info"]["supports_reasoning"])
        self.assertEqual(body["model_info"]["max_input_tokens"], extra.ARGON_INPUT_LIMIT)

    def test_an_absent_upstream_plans_nothing(self):
        self.assertEqual(extra.plan_argon([], None), ("absent", None))


class AnthropicTest(unittest.TestCase):
    MODELS = [{
        "id": "claude-x", "lifecycle": "active",
        "max_input_tokens": 500000, "max_tokens": 100000,
    }]

    def test_an_active_model_is_created_on_the_official_route(self):
        planned = extra.plan_anthropic(self.MODELS, [], {}, set())
        action, body = planned[0]
        self.assertEqual(action, "create")
        self.assertEqual(body["litellm_params"]["model"], "anthropic/claude-x")
        self.assertEqual(body["litellm_params"]["api_base"], extra.ANTHROPIC_API)
        self.assertEqual(body["model_info"]["max_output_tokens"], 100000)
        self.assertEqual(body["litellm_params"]["additional_drop_params"], ["prompt_cache_key"])

    def test_inactive_or_retired_models_are_skipped(self):
        models = [
            {**self.MODELS[0], "id": "claude-deprecated", "lifecycle": "deprecated"},
            self.MODELS[0],
        ]
        self.assertEqual(extra.plan_anthropic(models, [], {}, {"claude-x"}), [])

    def test_a_manual_route_is_preserved(self):
        rows = [{"model_name": "claude-x",
                 "litellm_params": {"litellm_credential_name": extra.ANTHROPIC_CREDENTIAL},
                 "model_info": {}}]
        self.assertEqual(extra.plan_anthropic(self.MODELS, rows, {}, set()), [])


class CursorCloudTest(unittest.TestCase):
    def test_a_snapshot_is_written_with_added_and_removed(self):
        with tempfile.TemporaryDirectory() as home:
            home = Path(home)
            (home / "cursor-cloud-model-catalog.json").write_text(
                json.dumps({"items": [{"id": "old"}]})
            )
            result = extra.sync_cursor_cloud(FakeClient(), home)
            self.assertEqual(result["models"], 1)
            self.assertEqual(result["added"], ["cloud-a"])
            self.assertEqual(result["removed"], ["old"])
            self.assertTrue((home / "cursor-cloud-model-catalog.json").exists())



class LMStudioTest(unittest.TestCase):
    MODELS = [
        {
            "id": "google/gemma-4-e4b",
            "type": "vlm",
            "max_context_length": 131072,
            "capabilities": ["tool_use"],
        },
        {
            "id": "text-embedding-nomic-embed-text-v1.5",
            "type": "embeddings",
            "max_context_length": 2048,
        },
    ]

    def test_plans_chat_vlm_and_embedding_models_with_zero_cost(self):
        planned = extra.plan_lmstudio(
            self.MODELS, [], "http://172.21.0.1:12340/v1"
        )
        self.assertEqual(len(planned), 2)
        a0, b0 = planned[0]
        self.assertEqual(a0, "create")
        self.assertEqual(b0["model_name"], "gemma-4-e4b")
        self.assertEqual(b0["litellm_params"]["model"], "lm_studio/google/gemma-4-e4b")
        self.assertEqual(b0["litellm_params"]["custom_llm_provider"], "lm_studio")
        self.assertEqual(b0["litellm_params"]["litellm_credential_name"], extra.LMSTUDIO_CREDENTIAL)
        self.assertEqual(b0["litellm_params"]["api_base"], "http://172.21.0.1:12340/v1")
        self.assertEqual(b0["model_info"]["mode"], "chat")
        self.assertEqual(b0["model_info"]["max_input_tokens"], 131072)
        self.assertEqual(b0["litellm_params"]["input_cost_per_token"], 0.0)
        self.assertEqual(b0["litellm_params"]["output_cost_per_token"], 0.0)
        self.assertTrue(b0["model_info"]["supports_vision"])
        self.assertTrue(b0["model_info"]["supports_function_calling"])

        a1, b1 = planned[1]
        self.assertEqual(a1, "create")
        self.assertEqual(b1["model_name"], "text-embedding-nomic-embed-text-v1.5")
        self.assertEqual(b1["model_info"]["mode"], "embedding")
        self.assertEqual(b1["model_info"]["max_input_tokens"], 2048)
        self.assertEqual(b1["litellm_params"]["input_cost_per_token"], 0.0)
        self.assertEqual(b1["litellm_params"]["output_cost_per_token"], 0.0)

    def test_colliding_tails_are_disambiguated_without_slash(self):
        models = [
            {"id": "org-a/shared-model", "type": "llm"},
            {"id": "org-b/shared-model", "type": "llm"},
        ]
        planned = extra.plan_lmstudio(models, [], "http://172.21.0.1:12340/v1")
        names = [body["model_name"] for _, body in planned]
        self.assertEqual(names, ["org-a-shared-model", "org-b-shared-model"])

    def test_discovery_timeout_preserves_other_provider_results(self):
        with tempfile.TemporaryDirectory() as home, mock.patch.multiple(
            extra, sync_cursor_cloud=mock.DEFAULT, sync_cursor_chat=mock.DEFAULT,
            sync_anthropic=mock.DEFAULT, sync_argon=mock.DEFAULT,
            litellm_container=mock.DEFAULT,
        ) as mocks, mock.patch.object(
            extra.subprocess, "run", side_effect=subprocess.TimeoutExpired("docker", 60)
        ):
            mocks["sync_argon"].return_value = {"action": "unchanged"}
            result = extra.sync_all(FakeClient(), Path(home), [])
        self.assertEqual(result["argon"], {"action": "unchanged"})
        self.assertEqual(result["lmstudio"], {"error": "RuntimeError: LM Studio catalog discovery timed out"})

    def test_transformed_names_avoid_registry_and_plan_collisions(self):
        models = [
            {"id": identifier, "type": "llm"}
            for identifier in ("a/b-c/m", "a-b/c/m", "a-b-c-m")
        ]
        existing = [{"model_name": "a-b-c-m", "litellm_params": {}, "model_info": {}}]
        planned = extra.plan_lmstudio(models, existing, "http://example.test/v1")
        names = {body["model_name"] for _, body in planned}
        self.assertEqual(len(names), 3)
        self.assertNotIn("a-b-c-m", {body["model_name"] for _, body in planned[:2]})
        self.assertTrue(all("/" not in name for name in names))

    def test_owned_name_survives_catalog_changes_and_duplicates_fail_closed(self):
        api_base = "http://example.test/v1"
        models = [{"id": f"{org}/shared", "type": "llm"} for org in ("a", "b")]
        _, first = extra.plan_lmstudio(models[:1], [], api_base)[0]
        first["model_info"]["id"] = "existing"
        planned = extra.plan_lmstudio(models, [first], api_base)
        self.assertEqual([(action, body["model_name"]) for action, body in planned],
                         [("unchanged", "shared"), ("create", "b-shared")])
        duplicate = {**first, "model_name": "a-shared"}
        with self.assertRaisesRegex(ValueError, "ambiguous LM Studio route"):
            extra.plan_lmstudio(models, [first, duplicate], api_base)

    def test_model_type_takes_precedence_and_unknown_models_are_skipped(self):
        models = [
            {"id": "embed-chat", "type": "llm"},
            {"id": "embed-vision", "type": "vlm"},
            {"id": "no_name_hint", "type": "embedding"},
            {"id": "embed-fallback"},
            {"id": "bge-small"},
            {"id": "embed-unknown", "type": "other"},
        ]
        planned = extra.plan_lmstudio(models, [], "http://example.test/v1")
        self.assertEqual(
            {body["model_name"]: body["model_info"]["mode"] for _, body in planned},
            {"embed-chat": "chat", "embed-vision": "chat",
             "no_name_hint": "embedding", "embed-fallback": "embedding"},
        )

    def test_manual_route_is_preserved_and_owned_route_is_idempotent(self):
        manual_rows = [{
            "model_name": "gemma-4-e4b",
            "litellm_params": {"litellm_credential_name": extra.LMSTUDIO_CREDENTIAL},
            "model_info": {},
        }]
        planned = extra.plan_lmstudio([self.MODELS[0]], manual_rows, "http://172.21.0.1:12340/v1")
        self.assertEqual(planned, [])

        manual_rows[0]["litellm_params"]["model"] = "lm_studio/another/gemma-4-e4b"
        planned = extra.plan_lmstudio([self.MODELS[0]], manual_rows, "http://example.test/v1")
        self.assertEqual(planned[0][1]["model_name"], "google-gemma-4-e4b")

        created = extra.plan_lmstudio([self.MODELS[0]], [], "http://172.21.0.1:12340/v1")[0][1]
        existing = [{
            "model_name": created["model_name"],
            "litellm_params": dict(created["litellm_params"]),
            "model_info": {**created["model_info"], "id": "lm-dep-1"},
        }]
        replan = extra.plan_lmstudio([self.MODELS[0]], existing, "http://172.21.0.1:12340/v1")
        self.assertEqual(replan[0][0], "unchanged")

    def test_joins_existing_other_provider_group_with_next_order(self):
        existing = [{
            "model_name": "gemma-4-e4b",
            "litellm_params": {"order": 2, "litellm_credential_name": "OpenRouter"},
            "model_info": {"id": "or-1"},
        }]
        action, body = extra.plan_lmstudio([self.MODELS[0]], existing, "http://172.21.0.1:12340/v1")[0]
        self.assertEqual(action, "create")
        self.assertEqual(body["litellm_params"]["order"], 3)



class UnusedCredentialsTest(unittest.TestCase):
    def test_unused_credential_selection_skips_dedicated_manual_and_cursor(self):
        creds = [
            {"credential_name": "Nvidia Nim SUB", "credential_values": {"custom_llm_provider": "NVIDIA_NIM"}},
            {"credential_name": "lmstudio", "credential_values": {"custom_llm_provider": "lm_studio"}},
            {"credential_name": "cursor-api", "credential_values": {"custom_llm_provider": "cursor"}},
            {"credential_name": "OpenAI-API-blocked", "credential_values": {"custom_llm_provider": "OpenAI"}},
            {"credential_name": "new-owned", "credential_values": {"custom_llm_provider": "OpenAI"}},
        ]
        rows = [
            {
                "model_name": "gpt-6",
                "litellm_params": {"litellm_credential_name": "OpenAI-API-blocked"},
                "model_info": {"blocked": True},
            },
            {
                "model_name": "gpt-new",
                "litellm_params": {"litellm_credential_name": "new-owned"},
                "model_info": {"credential_auto_sync": extra.CREDENTIAL_OWNER},
            },
        ]
        self.assertEqual(
            extra.unused_credential_names(creds, rows),
            ["Nvidia Nim SUB", "new-owned"],
        )

    def test_nvidia_nim_planning_uses_hosted_vllm_filters_noise_and_handles_null_order(self):
        models = [
            {"id": "nvidia/nemotron-3.5-bolt-e4b", "mode": "chat"},
            {"id": "nvidia/llama-3.1-nemoguard-8b-topic-control", "mode": "chat"},
            {"id": "fastino/gliner2-pii", "mode": "chat"},
            {"id": "nvidia/ai-synthetic-video-detector", "mode": "chat"},
            {"id": "nvidia/nemotron-3.5-embed-1b", "mode": "embedding"},
        ]
        existing = [{
            "model_name": "nemotron-3.5-bolt-e4b",
            "litellm_params": {"litellm_credential_name": "nvidia", "order": None},
            "model_info": {"id": "nim-manual-1"},
        }]
        planned = extra.plan_credential_models(
            "Nvidia Nim SUB", "NVIDIA_NIM", "https://integrate.api.nvidia.com/v1",
            models, existing,
        )
        names = [b["model_name"] for _, b in planned]
        self.assertEqual(names, ["nemotron-3.5-bolt-e4b", "nemotron-3.5-embed-1b"])
        a0, b0 = planned[0]
        self.assertEqual(a0, "create")
        self.assertEqual(b0["litellm_params"]["model"], "hosted_vllm/nvidia/nemotron-3.5-bolt-e4b")
        self.assertEqual(b0["litellm_params"]["custom_llm_provider"], "hosted_vllm")
        self.assertEqual(b0["litellm_params"]["order"], 2)
        self.assertEqual(b0["model_info"]["credential_auto_sync"], extra.CREDENTIAL_OWNER)

        # Idempotent on re-run
        existing_after = existing + [{
            "model_name": b0["model_name"],
            "litellm_params": dict(b0["litellm_params"]),
            "model_info": {**b0["model_info"], "id": "nim-sub-1"},
        }]
        replan = extra.plan_credential_models(
            "Nvidia Nim SUB", "NVIDIA_NIM", "https://integrate.api.nvidia.com/v1",
            [models[0]], existing_after,
        )
        self.assertEqual(replan[0][0], "unchanged")

    def test_dated_snapshots_are_skipped_when_base_alias_exists(self):
        models = [
            {"id": "gpt-5.4", "mode": "chat"},
            {"id": "gpt-5.4-2026-03-05", "mode": "chat"},
            {"id": "gpt-realtime-mini", "mode": "chat"},
        ]
        planned = extra.plan_credential_models("openai-new", "OpenAI", "", models, [])
        self.assertEqual([b["model_name"] for _, b in planned], ["gpt-5.4"])

    def test_sync_probes_and_reverts_unhealthy_with_cooldown_and_short_circuits_budget(self):
        class CredClient(FakeClient):
            def __init__(self):
                super().__init__()
                self.seq = 0

            def get(self, path, timeout=None):
                self.calls.append(("GET", path))
                if path == "/credentials":
                    return {"credentials": [
                        {"credential_name": "sub-ok", "credential_values": {"custom_llm_provider": "NVIDIA_NIM"}},
                        {"credential_name": "sub-broke", "credential_values": {"custom_llm_provider": "OpenAI"}},
                    ]}
                return super().get(path, timeout=timeout)

            def post(self, path, body, timeout=None):
                self.calls.append(("POST", path, body))
                if path == "/model/new":
                    self.seq += 1
                    return {"model_id": f"dep-{self.seq}"}
                return {"ok": True}

        discovered = [
            {
                "credential": "sub-ok",
                "provider": "NVIDIA_NIM",
                "api_base": "https://integrate.api.nvidia.com/v1",
                "models": [
                    {"id": "nvidia/good-model", "mode": "chat"},
                    {"id": "nvidia/bad-model", "mode": "chat"},
                ],
            },
            {
                "credential": "sub-broke",
                "provider": "OpenAI",
                "api_base": "https://api.openai.com/v1",
                "models": [
                    {"id": "gpt-a", "mode": "chat"},
                    {"id": "gpt-b", "mode": "chat"},
                ],
            },
        ]

        def fake_probe(_client, model_id):
            if model_id == "dep-1":
                return "unhealthy"
            if model_id == "dep-2":
                return "ok"
            return "budget_exceeded"

        with tempfile.TemporaryDirectory() as home_str:
            home = Path(home_str)
            client = CredClient()
            report = extra.sync_unused_credentials(
                client, home, [], apply=True,
                probe=fake_probe,
                discover=lambda targets: [d for d in discovered if d["credential"] in targets],
                now=1000.0,
            )
            self.assertEqual(report["created"], 1)
            self.assertEqual(report["reverted"], 2)
            # dep-1 (bad-model) and dep-3 (gpt-a, budget_exceeded) deleted; gpt-b skipped via short-circuit
            deletes = [c[2]["id"] for c in client.calls if c[0] == "POST" and c[1] == "/model/delete"]
            self.assertEqual(deletes, ["dep-1", "dep-3"])
            self.assertEqual(report["credentials"]["sub-ok"]["kept_models"], ["good-model"])

            # Second run within 24h skips cooled-down bad-model and sub-broke credential
            client2 = CredClient()
            report2 = extra.sync_unused_credentials(
                client2, home, [], apply=False,
                probe=fake_probe,
                discover=lambda targets: [d for d in discovered if d["credential"] in targets],
                now=1100.0,
            )
            self.assertEqual(report2["cooldown_credentials"], ["sub-broke"])
            self.assertEqual(report2["credentials"]["sub-ok"]["models"], ["good-model"])

if __name__ == "__main__":
    unittest.main()
