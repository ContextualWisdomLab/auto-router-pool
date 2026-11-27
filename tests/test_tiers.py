import unittest

from auto_router_pool import tiers


class RoutersTest(unittest.TestCase):
    def setUp(self):
        self.routers = tiers.routers(
            "primary", "premium", "budget", "runner-up", ("micro-a", "micro-b")
        )

    def test_expected_tier_counts(self):
        # Media tiers are absent until a deployment of that modality exists.
        self.assertEqual(len(self.routers["auto"]), 6)
        self.assertEqual(len(self.routers["auto-classify"]), 4)
        self.assertEqual(len(self.routers["auto-strix"]), 4)
        self.assertEqual(len(self.routers["auto-advisor"]), 4)

    def test_no_router_exceeds_the_engine_cap(self):
        for name, tier_map in self.routers.items():
            self.assertGreaterEqual(len(tier_map), tiers.MIN_TIERS, name)
            self.assertLessEqual(len(tier_map), tiers.MAX_TIERS, name)

    def test_every_published_tier_has_a_model(self):
        for tier_map in self.routers.values():
            for tier, entries in tier_map.items():
                self.assertTrue(entries, tier)

    def test_every_auto_tier_is_described_exactly_once(self):
        published = set(self.routers["auto"])
        self.assertTrue(published <= set(tiers.AUTO))
        self.assertEqual(len(tiers.AUTO), len(set(tiers.AUTO)))
        for tier in published:
            self.assertIn(tier, tiers.TIER_PROFILE)

    def test_definitions_examples_and_profiles_cover_the_taxonomy(self):
        # MECE at the contract level: every shipped name is defined exactly once
        # with its own text, has its own example, and is priced by a profile.
        self.assertEqual(set(tiers.AUTO), set(tiers.AUTO_DEFINITIONS))
        self.assertEqual(set(tiers.AUTO), set(tiers.AUTO_EXAMPLES))
        self.assertEqual(set(tiers.AUTO), set(tiers.TIER_PROFILE))
        self.assertEqual(
            len(set(tiers.AUTO_DEFINITIONS.values())), len(tiers.AUTO_DEFINITIONS)
        )

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
        frontier = self.routers["auto"][tiers.FRONTIER]
        self.assertEqual(frontier[0], ("premium", "xhigh"))

    def test_light_tier_does_not_use_low_effort_for_budget(self):
        simple = self.routers["auto"][tiers.T_SIMPLE]
        self.assertEqual(simple[0], ("budget", "medium"))

    def test_contract_covers_exactly_the_published_tiers(self):
        built = tiers.policies(self.routers)
        defined = [item["name"] for item in built["auto"]["tier_definitions"]]
        self.assertEqual(defined, list(self.routers["auto"]))
        self.assertIn(built["auto"]["fallback_tier"], defined)
        self.assertTrue(built["auto"]["classification_examples"])

    def test_focus_routers_do_not_define_their_own_tiers(self):
        # The engine registers one custom-tier router; that is ``auto``, so the
        # focus routers must not carry definitions or a private prompt.
        built = tiers.policies(self.routers)
        for name in ("auto-strix", "auto-advisor"):
            self.assertNotIn("tier_definitions", built[name])
            self.assertNotIn("classification_prompt", built[name])

    def test_every_router_classifies_every_request(self):
        for name, settings in tiers.policies(self.routers).items():
            self.assertEqual(settings["classification_mode"], "every_request", name)
            self.assertFalse(settings["session_affinity"], name)

    def test_free_models_do_not_reorder_the_classification_group(self):
        # Classification is judged on label stability, not price, so a free
        # deployment must not lead its model group there.
        ladders = {
            "primary": {"high": {"index": 55.0, "in": 3, "out": 15}},
            "runner": {"high": {"index": 50.0, "in": 3, "out": 15}},
            "free": {"high": {"index": 52.0, "in": 3, "out": 15}},
        }
        plain = tiers.routers(
            "primary", "premium", "budget", "runner", ("micro",), ladders=ladders
        )
        boosted = tiers.routers(
            "primary", "premium", "budget", "runner", ("micro",),
            ladders=ladders, free_models={"free"},
        )
        self.assertEqual(plain["auto-classify"], boosted["auto-classify"])
        self.assertTrue(
            any(chain and chain[0][0] == "free" for chain in boosted["auto"].values())
        )

    def test_focus_routers_use_the_shipped_tiers(self):
        for name in ("auto-classify", "auto-strix", "auto-advisor"):
            self.assertEqual(tuple(self.routers[name]), tiers.SHIPPED, name)

    def test_security_keeps_a_code_literate_light_tier(self):
        simple = self.routers["auto-strix"]["SIMPLE"]
        self.assertEqual(simple[0], ("budget", "medium"))
        self.assertIn(("primary", "low"), simple)

    def test_monitoring_starts_on_the_budget_model(self):
        self.assertEqual(self.routers["auto-advisor"]["SIMPLE"][0], ("budget", "medium"))
        self.assertEqual(self.routers["auto-advisor"]["COMPLEX"][0], ("primary", "medium"))

    def test_top_tier_uses_each_models_peak_effort(self):
        routers = tiers.routers(
            "primary", "premium", "budget", "runner-up", (),
            {"premium": "xhigh", "primary": "max", "runner-up": "max"},
        )
        frontier = routers["auto"][tiers.FRONTIER]
        self.assertEqual(
            frontier,
            [("premium", "xhigh"), ("primary", "max"), ("runner-up", "max")],
        )
        self.assertEqual(
            routers["auto-strix"]["REASONING"][0], ("premium", "xhigh")
        )


class MediaTierTest(unittest.TestCase):
    def routers(self, specialized):
        return tiers.routers(
            "primary", "premium", "budget", "runner-up", (),
            specialized=specialized,
        )["auto"]

    def test_media_tiers_hold_only_their_own_modality(self):
        built = self.routers({"image": ["painter"], "video": ["filmer"]})
        self.assertEqual(built[tiers.T_IMAGE], [("painter", None)])
        self.assertEqual(built[tiers.T_VIDEO], [("filmer", None)])

    def test_text_tiers_never_fall_back_to_a_generator(self):
        built = self.routers({"image": ["painter"], "video": ["filmer"]})
        for tier, entries in built.items():
            if tier in tiers.MEDIA_TIERS:
                continue
            self.assertNotIn("painter", [model for model, _ in entries], tier)
            self.assertNotIn("filmer", [model for model, _ in entries], tier)

    def test_media_tiers_disappear_without_a_deployment(self):
        built = self.routers({"image": [], "video": []})
        self.assertNotIn(tiers.T_IMAGE, built)
        self.assertNotIn(tiers.T_VIDEO, built)

    def test_media_entries_carry_no_effort(self):
        built = self.routers({"image": ["painter"], "video": []})
        _, configs = tiers.build(built)
        self.assertEqual(configs[tiers.T_IMAGE][0]["litellm_params"], {})


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
