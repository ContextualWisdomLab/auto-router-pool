import json
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
