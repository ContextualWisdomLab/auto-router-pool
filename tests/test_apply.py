import unittest

from auto_router_pool import apply

TIERS = {"SIMPLE": ["small", "large"]}


def model(name, cap, tiers=None):
    return {
        "model_name": name,
        "model_info": {"id": name + "-id", "max_input_tokens": cap},
        "litellm_params": {
            "complexity_router_config": {"tiers": tiers or {}, "tier_model_configs": {}}
        },
    }


def router(name, cap, tier_map):
    tiers, configs = apply.build(tier_map)
    row = model(name, cap, tiers)
    row["litellm_params"]["complexity_router_config"]["tier_model_configs"] = configs
    return row


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.patches = []

    def patch(self, path, body):
        self.patches.append(body)
        ident = path.split("/")[2]
        for row in self.rows:
            if row["model_info"]["id"] == ident:
                row["model_info"].update(body.get("model_info") or {})
                row["litellm_params"].update(body.get("litellm_params") or {})

    def get(self, path):
        return {"data": self.rows}


class FrozenClient(FakeClient):
    def patch(self, path, body):
        self.patches.append(body)


class ContextCapTest(unittest.TestCase):
    def test_smallest_solver_window_wins(self):
        rows = [model("router", 500000, TIERS), model("small", 131072), model("large", 400000)]
        self.assertEqual(apply.context_cap(rows, TIERS), 131072)

    def test_undeclared_window_proves_no_floor(self):
        rows = [model("router", 500000, TIERS), model("small", None), model("large", 400000)]
        self.assertIsNone(apply.context_cap(rows, TIERS))

    def test_empty_tiers_have_no_floor(self):
        self.assertIsNone(apply.context_cap([model("router", 500000)], {}))

    def test_drift_lists_routers_above_their_tiers(self):
        rows = [
            router("auto", 500000, {"SIMPLE": [("small", "low")]}),
            model("small", 131072),
            router("auto-advisor", 400000, {"SIMPLE": [("large", "high")]}),
            model("large", 400000),
        ]
        self.assertEqual(apply.context_drift(rows, ("auto", "auto-advisor")), ["auto"])
        self.assertEqual(apply.context_drift(rows), ["auto"])


class PlanTest(unittest.TestCase):
    def test_cap_only_change_leaves_params_alone(self):
        rows = [router("auto", 500000, {"SIMPLE": [("small", "low")]}), model("small", 131072)]
        changes = apply.plan(rows, {"auto": {"SIMPLE": [("small", "low")]}}, "small")
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["model_info"], {"max_input_tokens": 131072})
        self.assertNotIn("litellm_params", changes[0])
        self.assertEqual(changes[0]["previous_max_input_tokens"], 500000)

    def test_matching_cap_is_unchanged(self):
        rows = [router("auto", 131072, {"SIMPLE": [("small", "low")]}), model("small", 131072)]
        self.assertEqual(apply.plan(rows, {"auto": {"SIMPLE": [("small", "low")]}}, "small"), [])


class ClassifierTest(unittest.TestCase):
    def row_with_classifier(self, model_name, timeout, tiers=None):
        row = router("auto", 400000, tiers or {"SIMPLE": [("small", "low")], "MEDIUM": [("budget", "high")]})
        row["litellm_params"]["complexity_router_config"]["classifier_llm_config"] = {
            "model": model_name,
            "timeout_ms": timeout,
        }
        return row

    def test_plan_pins_classifier_to_budget(self):
        rows = [self.row_with_classifier("luna", 10000), model("small", 400000), model("budget", 400000)]
        changes = apply.plan(rows, {"auto": {"SIMPLE": [("small", "low")], "MEDIUM": [("budget", "high")]}}, "small", "budget")
        self.assertEqual(len(changes), 1)
        classifier = changes[0]["config"]["classifier_llm_config"]
        self.assertEqual(classifier["model"], "budget")
        self.assertEqual(classifier["timeout_ms"], apply.CLASSIFIER_TIMEOUT_MS)

    def test_pinned_classifier_is_unchanged(self):
        rows = [self.row_with_classifier("budget", apply.CLASSIFIER_TIMEOUT_MS),
                model("small", 400000), model("budget", 400000)]
        self.assertEqual(
            apply.plan(rows, {"auto": {"SIMPLE": [("small", "low")], "MEDIUM": [("budget", "high")]}}, "small", "budget"),
            [],
        )

    def test_classifier_drift_flags_foreign_model(self):
        rows = [self.row_with_classifier("luna", 30000)]
        self.assertEqual(apply.classifier_drift(rows), ["auto"])

    def test_classifier_drift_flags_short_timeout(self):
        rows = [self.row_with_classifier("small", 10000)]
        self.assertEqual(apply.classifier_drift(rows), ["auto"])

    def test_classifier_drift_ignores_matching_config(self):
        rows = [self.row_with_classifier("small", apply.CLASSIFIER_TIMEOUT_MS)]
        self.assertEqual(apply.classifier_drift(rows), [])

    def test_classifier_change_is_verified(self):
        rows = [self.row_with_classifier("luna", 10000), model("small", 400000), model("budget", 400000)]
        client = FakeClient(rows)
        changes = apply.plan(rows, {"auto": {"SIMPLE": [("small", "low")], "MEDIUM": [("budget", "high")]}}, "small", "budget")
        apply.apply(client, changes)
        classifier = rows[0]["litellm_params"]["complexity_router_config"]["classifier_llm_config"]
        self.assertEqual(classifier["model"], "budget")


class ApplyTest(unittest.TestCase):
    def test_apply_patches_cap_and_verifies(self):
        rows = [router("auto", 500000, {"SIMPLE": [("small", "low")]}), model("small", 131072)]
        client = FakeClient(rows)
        changes = apply.plan(rows, {"auto": {"SIMPLE": [("small", "low")]}}, "small")
        apply.apply(client, changes)
        self.assertEqual(rows[0]["model_info"]["max_input_tokens"], 131072)
        self.assertEqual(client.patches, [{"model_info": {"max_input_tokens": 131072}}])

    def test_verification_failure_rolls_back(self):
        rows = [router("auto", 500000, {"SIMPLE": [("small", "low")]}), model("small", 131072)]
        client = FrozenClient(rows)
        changes = apply.plan(rows, {"auto": {"SIMPLE": [("small", "low")]}}, "small")
        with self.assertRaises(RuntimeError):
            apply.apply(client, changes)
        self.assertEqual(
            client.patches,
            [
                {"model_info": {"max_input_tokens": 131072}},
                {"model_info": {"max_input_tokens": 500000}},
            ],
        )


class FallbackClient:
    def __init__(self, result):
        self.result = result
        self.posts = []

    def get(self, path):
        return self.result

    def post(self, path, body):
        self.posts.append((path, body))
        return {"ok": True}


class AutoFallbackTest(unittest.TestCase):
    def test_replaces_only_the_auto_edge(self):
        client = FallbackClient({"current_values": {"fallbacks": [
            {"auto": ["old"]}, {"other": ["keep"]}
        ]}})
        apply.set_auto_fallback(client, ["premium", "primary"])
        self.assertEqual(client.posts, [("/config/update", {
            "router_settings": {"fallbacks": [
                {"auto": ["premium", "primary"]}, {"other": ["keep"]}
            ]}
        })])

    def test_repeated_roles_collapse_to_first_mention(self):
        client = FallbackClient({"current_values": {"fallbacks": []}})
        apply.set_auto_fallback(
            client, ["luna", "luna", "lightning", "luna", None]
        )
        self.assertEqual(client.posts, [("/config/update", {
            "router_settings": {"fallbacks": [{"auto": ["luna", "lightning"]}]}
        })])

    def test_settings_error_leaves_fallbacks_untouched(self):
        client = FallbackClient({"error_status": "unreachable"})
        with self.assertRaises(RuntimeError):
            apply.set_auto_fallback(client, ["premium"])
        self.assertEqual(client.posts, [])


class ContractTest(unittest.TestCase):
    def test_contract_is_written_with_the_tiers(self):
        rows = [router("auto", 400000, {"SIMPLE": [("small", "low")]}), model("small", 400000)]
        contract = {
            "tier_definitions": [{"name": "SIMPLE", "description": "d"}],
            "fallback_tier": "SIMPLE",
        }
        changes = apply.plan(
            rows, {"auto": {"SIMPLE": [("small", "low")]}}, "small",
            extras={"auto": contract},
        )
        self.assertTrue(changes)
        self.assertEqual(changes[0]["config"]["fallback_tier"], "SIMPLE")

    def test_custom_tiers_drop_the_builtin_affinity_contract(self):
        rows = [router("auto", 400000, {"SIMPLE": [("small", "low")]}), model("small", 400000)]
        rows[0]["litellm_params"]["complexity_router_config"].update(
            {"session_affinity": True, "classifier_fallback": "default_model"}
        )
        changes = apply.plan(
            rows, {"auto": {"SIMPLE": [("small", "low")]}}, "small",
            extras={"auto": {"tier_definitions": [{"name": "SIMPLE", "description": "d"}]}},
        )
        config = changes[0]["config"]
        self.assertFalse(config["session_affinity"])
        self.assertNotIn("classifier_fallback", config)

    def test_matching_contract_needs_no_write(self):
        rows = [router("auto", 400000, {"SIMPLE": [("small", "low")]}), model("small", 400000)]
        contract = {"tier_definitions": [{"name": "SIMPLE", "description": "d"}]}
        rows[0]["litellm_params"]["complexity_router_config"].update(contract)
        self.assertEqual(
            apply.plan(
                rows, {"auto": {"SIMPLE": [("small", "low")]}}, "small",
                extras={"auto": contract},
            ),
            [],
        )


class MediaCapTest(unittest.TestCase):
    def test_media_deployments_do_not_bound_the_window(self):
        rows = [
            {"model_name": "chatty", "model_info": {"id": "c", "mode": "chat", "max_input_tokens": 400000}},
            {"model_name": "painter", "model_info": {"id": "p", "mode": "image_generation"}},
        ]
        self.assertEqual(apply.context_cap(rows, {"T": ["chatty", "painter"]}), 400000)


if __name__ == "__main__":
    unittest.main()


class DecisionClassifierPlanTest(unittest.TestCase):
    def row(self):
        row = router("auto", 400000, {"SIMPLE": [("small", "low")]})
        row["litellm_params"]["complexity_router_config"]["classifier_llm_config"] = {
            "model": "small", "timeout_ms": apply.CLASSIFIER_TIMEOUT_MS,
        }
        return row

    def decision(self, blocked=False):
        return {"model_name": "clef", "litellm_params": {},
                "model_info": {"id": "clef-1", "mode": "decision", "blocked": blocked}}

    def test_plan_writes_the_decision_classifier(self):
        rows = [self.row(), model("small", 400000), self.decision()]
        changes = apply.plan(
            rows, {"auto": {"SIMPLE": [("small", "low")]}}, "small",
            {"type": apply.DECISION_CLASSIFIER, "model": "clef"},
        )
        config = changes[0]["config"]
        self.assertEqual(config["classifier_type"], "oss_classifier")
        self.assertEqual(
            config["opensource_classifier_config"]["deployment_name"], "clef"
        )
        self.assertEqual(
            config["opensource_classifier_config"]["timeout_ms"],
            apply.DECISION_CLASSIFIER_TIMEOUT_MS,
        )

    def test_switching_back_to_the_llm_judge_changes_the_type(self):
        row = self.row()
        row["litellm_params"]["complexity_router_config"].update({
            "classifier_type": "oss_classifier",
            "opensource_classifier_config": {"deployment_name": "clef",
                                             "timeout_ms": 30000},
        })
        rows = [row, model("small", 400000)]
        changes = apply.plan(
            rows, {"auto": {"SIMPLE": [("small", "low")]}}, "small", "small"
        )
        config = changes[0]["config"]
        self.assertEqual(config["classifier_type"], "llm")
        self.assertEqual(config["classifier_llm_config"]["model"], "small")

    def test_a_blocked_decision_deployment_is_drift(self):
        row = self.row()
        row["litellm_params"]["complexity_router_config"].update({
            "classifier_type": "oss_classifier",
            "opensource_classifier_config": {"deployment_name": "clef",
                                             "timeout_ms": 30000},
        })
        self.assertEqual(apply.classifier_drift([row, self.decision(True)]), ["auto"])

    def test_an_llm_judge_is_drift_while_a_decision_model_is_usable(self):
        self.assertEqual(
            apply.classifier_drift([self.row(), self.decision()]), ["auto"]
        )

    def test_the_decision_form_is_verified_on_read_back(self):
        rows = [self.row(), model("small", 400000), self.decision()]
        client = FakeClient(rows)
        changes = apply.plan(
            rows, {"auto": {"SIMPLE": [("small", "low")]}}, "small",
            {"type": apply.DECISION_CLASSIFIER, "model": "clef"},
        )
        apply.apply(client, changes)
        config = rows[0]["litellm_params"]["complexity_router_config"]
        self.assertEqual(config["classifier_type"], "oss_classifier")
        self.assertEqual(config["opensource_classifier_config"]["deployment_name"], "clef")
