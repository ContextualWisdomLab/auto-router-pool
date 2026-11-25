import unittest

from auto_router_pool import tiers


class RoutersTest(unittest.TestCase):
    def setUp(self):
        self.routers = tiers.routers(
            "primary", "premium", "budget", "runner-up", ("micro-a", "micro-b")
        )

    def test_expected_tier_counts(self):
        self.assertEqual(len(self.routers["auto"]), 7)
        self.assertEqual(len(self.routers["auto-classify"]), 4)
        self.assertEqual(len(self.routers["auto-strix"]), 4)

    def test_no_empty_model_names(self):
        for tier_map in self.routers.values():
            for tier, entries in tier_map.items():
                for model, _ in entries:
                    self.assertTrue(model, tier)

    def test_no_duplicate_model_inside_a_tier(self):
        for tier_map in self.routers.values():
            for tier, entries in tier_map.items():
                names = [model for model, _ in entries]
                self.assertEqual(len(names), len(set(names)), tier)

    def test_efforts_are_known(self):
        allowed = {"low", "medium", "high", "xhigh", "max"}
        for tier_map in self.routers.values():
            for entries in tier_map.values():
                for _, effort in entries:
                    self.assertIn(effort, allowed)

    def test_frontier_starts_with_premium(self):
        frontier = self.routers["auto"]["최고난도 추론·심층 연구·복잡한 장기 작업"]
        self.assertEqual(frontier[0], ("premium", "xhigh"))

    def test_light_tier_does_not_use_low_effort_for_budget(self):
        simple = self.routers["auto"]["명시 정보 추출·단순 분류·고정 규칙 처리"]
        self.assertEqual(simple[0], ("budget", "medium"))

    def test_top_tier_uses_each_models_peak_effort(self):
        routers = tiers.routers(
            "primary", "premium", "budget", "runner-up", (),
            {"premium": "xhigh", "primary": "max", "runner-up": "max"},
        )
        frontier = routers["auto"]["최고난도 추론·심층 연구·복잡한 장기 작업"]
        self.assertEqual(
            frontier,
            [("premium", "xhigh"), ("primary", "max"), ("runner-up", "max")],
        )
        self.assertEqual(routers["auto-strix"]["REASONING"], [("primary", "max")])


class TopEffortTest(unittest.TestCase):
    def test_picks_highest_measured_index(self):
        self.assertEqual(
            tiers.top_effort({"xhigh": {"index": 50.0}, "max": {"index": 49.2}}),
            "xhigh",
        )
        self.assertEqual(
            tiers.top_effort({"xhigh": {"index": 23.5}, "max": {"index": 27.9}}),
            "max",
        )

    def test_empty_ladder_falls_back_to_xhigh(self):
        self.assertEqual(tiers.top_effort({}), "xhigh")


class BuildTest(unittest.TestCase):
    def test_build_matches_litellm_shape(self):
        tier_map = {"SIMPLE": [("a", "low"), ("b", "medium")]}
        built, configs = tiers.build(tier_map)
        self.assertEqual(built, {"SIMPLE": ["a", "b"]})
        self.assertEqual(
            configs["SIMPLE"][1],
            {"model_name": "b", "litellm_params": {"reasoning_effort": "medium"}},
        )

    def test_single_micro_model_still_builds(self):
        routers = tiers.routers("p", "m", "b", "r", ("only-one",))
        for entries in routers["auto"].values():
            self.assertTrue(all(name for name, _ in entries))


if __name__ == "__main__":
    unittest.main()
