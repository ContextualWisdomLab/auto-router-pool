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


if __name__ == "__main__":
    unittest.main()
