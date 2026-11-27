import unittest

from auto_router_pool import assign, tiers


def ladder(**pairs):
    return {effort: {"index": index, "in": 1, "out": 1} for effort, index in pairs.items()}


ASTRA = ladder(max=62.19, xhigh=61.76, high=60.29, medium=57.05, low=45.65)
SOL = ladder(max=57.26, xhigh=55.63, high=54.84, medium=52.43, low=40.55)
OLD_SOL = ladder(max=54.66, xhigh=52.41, high=50.28, medium=46.35, low=34.80)


class PlaceTest(unittest.TestCase):
    def setUp(self):
        self.ladders = {"astra": ASTRA, "sol": SOL, "old": OLD_SOL}

    def test_target_is_the_weakest_ceiling(self):
        placed = assign.place(["astra", "sol", "old"], self.ladders)
        self.assertEqual(placed["target"], 54.66)
        self.assertEqual(
            placed["entries"],
            [("astra", "medium", 57.05), ("sol", "high", 54.84), ("old", "max", 54.66)],
        )
        self.assertAlmostEqual(placed["spread"], 2.39, places=2)

    def test_tight_band_drops_the_model_that_cannot_fit(self):
        placed = assign.place(["astra", "sol", "old"], self.ladders, band=0.5)
        # astra and sol are 0.21 apart, so that pair fits a half-point band and
        # it assumes a higher score than the sol + old pair would.
        self.assertEqual(placed["target"], 57.05)
        self.assertEqual([model for model, _, _ in placed["entries"]], ["astra", "sol"])
        self.assertEqual(placed["skipped"], ["old"])
        self.assertLessEqual(placed["spread"], 0.5)

    def test_cheapest_rung_wins_inside_the_band(self):
        pricey = {
            "max": {"index": 60.0, "in": 9, "out": 36},
            "medium": {"index": 55.0, "in": 1, "out": 4},
        }
        placed = assign.place(
            ["pricey", "flat"], {"pricey": pricey, "flat": ladder(max=54.0)}, band=8.0
        )
        self.assertEqual(placed["entries"][0], ("pricey", "medium", 55.0))

    def test_equal_price_takes_the_lower_rung(self):
        flat = {
            "max": {"index": 60.0, "in": 1, "out": 1},
            "high": {"index": 56.0, "in": 1, "out": 1},
        }
        placed = assign.place(
            ["flat", "anchor"], {"flat": flat, "anchor": ladder(max=55.0)}, band=8.0
        )
        self.assertEqual(placed["entries"][0], ("flat", "high", 56.0))

    def test_frontier_chain_prefers_the_free_model(self):
        ladders = {
            "paid": {"max": {"index": 50.0, "in": 3, "out": 15}},
            "free": {"max": {"index": 50.0, "in": 3, "out": 15}},
        }
        placed = assign.place(["paid", "free"], ladders, free_models={"free"})
        self.assertEqual(placed["entries"][0][0], "free")

    def test_unrated_model_is_reported_not_placed(self):
        placed = assign.place(["astra", "sol", "ghost"], {"astra": ASTRA, "sol": SOL})
        self.assertEqual(placed["unrated"], ["ghost"])
        self.assertEqual(len(placed["entries"]), 2)

    def test_single_rated_model_is_left_alone(self):
        placed = assign.place(["astra", "ghost"], {"astra": ASTRA})
        self.assertIsNone(placed["target"])
        self.assertEqual(placed["entries"], [])


class FrontierTest(unittest.TestCase):
    def test_frontier_is_placed_for_equal_score(self):
        routers = tiers.routers(
            "sol", "astra", "budget", "old", (), {},
            ladders={"astra": ASTRA, "sol": SOL, "old": OLD_SOL},
        )
        frontier = routers["auto"]["최고난도 추론·심층 연구·복잡한 장기 작업"]
        self.assertEqual(frontier, [("astra", "medium"), ("sol", "high"), ("old", "max")])

    def test_routers_pad_every_tier_when_ladders_are_known(self):
        routers = tiers.routers(
            "sol", "astra", "budget", "old", (), {},
            ladders={"astra": ASTRA, "sol": SOL, "old": OLD_SOL, "budget": ladder(max=18.5)},
        )
        for tier_map in routers.values():
            for tier, entries in tier_map.items():
                # Width is bounded by the roster, never by a configured number.
                self.assertLessEqual(len(entries), 4, tier)
                names = [model for model, _ in entries]
                self.assertEqual(len(names), len(set(names)), tier)
        vision = routers["auto"][tiers.T_VISION]
        self.assertEqual(len(vision), 3)

    def test_frontier_keeps_peak_efforts_without_ladders(self):
        routers = tiers.routers("sol", "astra", "budget", "old", (), {"astra": "max"})
        frontier = routers["auto"]["최고난도 추론·심층 연구·복잡한 장기 작업"]
        self.assertEqual(frontier, [("astra", "max"), ("sol", "xhigh"), ("old", "xhigh")])


class RolesTest(unittest.TestCase):
    def setUp(self):
        def priced(pairs, cost):
            return {
                effort: {"index": index, "in": cost[0], "out": cost[1]}
                for effort, index in pairs.items()
            }

        self.ladders = {
            "astra": ASTRA, "sol": SOL, "old": OLD_SOL,
            "budget": priced(
                {"low": 9.7, "medium": 18.5, "high": 21.4, "max": 27.9}, (0.1, 0.5)
            ),
            "opus": priced(
                {"medium": 48.26, "high": 51.79, "xhigh": 57.52, "max": 59.12},
                (4, 20),
            ),
        }

    def test_solves_each_role_without_a_price_cap(self):
        chosen = assign.roles(self.ladders)
        self.assertEqual(chosen["premium"][0], "astra")
        self.assertEqual(chosen["premium"][3]["effort"], "xhigh")
        self.assertEqual(chosen["primary"][0], "sol")
        self.assertEqual(chosen["runner_up"][0], "old")
        self.assertEqual(chosen["budget"][0], "budget")

    def test_roles_never_reuse_a_model(self):
        chosen = assign.roles(self.ladders)
        names = [chosen[role][0] for role in ("premium", "primary", "runner_up", "budget")]
        self.assertEqual(len(names), len(set(names)))

    def test_cheapest_model_inside_the_floor_wins(self):
        cheap = {"max": {"index": 56.5, "in": 1, "out": 2}}
        dear = {"max": {"index": 57.0, "in": 9, "out": 36}}
        chosen = assign.roles({"cheap": cheap, "dear": dear, "other": ladder(max=50.0)})
        # Both reach the premium floor; the cheaper one takes the role.
        self.assertEqual(chosen["premium"][0], "cheap")

    def test_single_effort_ladder_still_competes(self):
        chosen = assign.roles({"only": ladder(low=9.0)})
        self.assertEqual(chosen["premium"][0], "only")
        self.assertEqual(chosen["budget"][0], "only")

    def test_empty_roster_assigns_nothing(self):
        self.assertEqual(set(assign.roles({}).values()), {None})


class CostTest(unittest.TestCase):
    def test_input_and_output_prices_are_priced_separately(self):
        # Cheap prompt with a dear answer versus the reverse, same index.
        chatty = {"index": 50.0, "in": 0.1, "out": 30.0}
        bulk = {"index": 50.0, "in": 5.0, "out": 0.1}
        # A prompt-heavy run prefers the cheap prompt, an answer-heavy run the
        # cheap answer: one blended number could only pick one of them.
        self.assertLess(assign.cost(chatty, "extract"), assign.cost(bulk, "extract"))
        self.assertLess(assign.cost(bulk, "deep"), assign.cost(chatty, "deep"))

    def test_profile_decides_the_cheapest_rung(self):
        # Same two rungs, opposite winners depending on the token mix.
        rungs = assign._rungs({
            "max": {"index": 60.0, "in": 0.1, "out": 40.0},
            "high": {"index": 58.0, "in": 6.0, "out": 0.2},
        })
        self.assertEqual(assign._cheapest(rungs, 58.0, 3.0, "extract")["effort"], "max")
        self.assertEqual(assign._cheapest(rungs, 58.0, 3.0, "deep")["effort"], "high")


class GrowTest(unittest.TestCase):
    def test_pads_a_tier_with_same_score_fallbacks(self):
        tier_map = {"frontier": [("old", "max")]}
        grown = assign.grow(
            tier_map, {"astra": ASTRA, "sol": SOL, "old": OLD_SOL}, limit=4
        )
        # Equal price, so the higher measured index comes first.
        self.assertEqual(
            grown["frontier"],
            [("old", "max"), ("astra", "medium"), ("sol", "high")],
        )

    def test_tier_width_comes_from_the_roster(self):
        # Every model whose ladder reaches the tier score inside the band joins,
        # so the size is a result of the constraints rather than a parameter.
        ladders = {"astra": ASTRA, "sol": SOL, "old": OLD_SOL, "anchor": ladder(max=55.0)}
        grown = assign.grow({"frontier": [("old", "max")]}, ladders)
        self.assertEqual(
            [model for model, _ in grown["frontier"]],
            ["old", "astra", "anchor", "sol"],
        )

    def test_models_below_the_tier_score_are_not_added(self):
        tier_map = {"frontier": [("astra", "max")]}
        grown = assign.grow(
            tier_map, {"astra": ASTRA, "sol": SOL, "old": OLD_SOL}, limit=4
        )
        self.assertEqual(grown["frontier"], [("astra", "max")])

    def test_limit_is_never_exceeded(self):
        tier_map = {"tier": [("old", "max")]}
        grown = assign.grow(
            tier_map, {"astra": ASTRA, "sol": SOL, "old": OLD_SOL}, limit=2
        )
        self.assertEqual(grown["tier"], [("old", "max"), ("astra", "medium")])

    def test_a_free_model_leads_the_group(self):
        # The first entry is the one the router tries, so a model the catalogue
        # prices at nothing moves ahead of an equally scored paid one.
        ladders = {
            "paid": {"max": {"index": 50.0, "in": 3, "out": 15}},
            "free": {"max": {"index": 50.0, "in": 3, "out": 15}},
        }
        grown = assign.grow(
            {"frontier": [("paid", "max")]}, ladders, free_models={"free"}
        )
        self.assertEqual(grown["frontier"], [("free", "max"), ("paid", "max")])

    def test_the_boost_only_reorders_the_same_members(self):
        ladders = {
            "paid": {"max": {"index": 50.0, "in": 3, "out": 15}},
            "free": {"max": {"index": 50.0, "in": 3, "out": 15}},
        }
        plain = assign.grow({"frontier": [("paid", "max")]}, ladders)
        boosted = assign.grow(
            {"frontier": [("paid", "max")]}, ladders, free_models={"free"}
        )
        self.assertEqual(set(plain["frontier"]), set(boosted["frontier"]))
        self.assertEqual(plain["frontier"][0], ("paid", "max"))

    def test_padding_cannot_shrink_the_window(self):
        tier_map = {"tier": [("old", "max")]}
        grown = assign.grow(
            tier_map,
            {"astra": ASTRA, "sol": SOL, "old": OLD_SOL},
            limit=4,
            windows={"old": 400000, "sol": 272000, "astra": 1000000},
        )
        self.assertEqual([model for model, _ in grown["tier"]], ["old", "astra"])


class ScheduleTest(unittest.TestCase):
    def test_every_level_is_covered_within_one_band(self):
        ladders = {"astra": ASTRA, "sol": SOL, "old": OLD_SOL}
        tiers = assign.schedule(["astra", "sol", "old"], ladders, band=3.0)
        levels = sorted({r["index"] for ladder_map in ladders.values() for r in assign._rungs(ladder_map)})
        for level in levels:
            hits = [t for t in tiers if t["target"] <= level <= t["covered"]]
            self.assertEqual(len(hits), 1, level)
        for tier in tiers:
            indexes = [index for _, _, index in tier["entries"]]
            self.assertLessEqual(max(indexes) - min(indexes), 3.0)
        # The range cannot be covered by fewer bands of this width.
        self.assertGreaterEqual(len(tiers), 6)

    def test_repeated_entry_sets_are_merged(self):
        tiers = assign.schedule(["wide"], {"wide": ladder(max=60.0, high=59.0)}, band=3.0)
        self.assertEqual([tier["target"] for tier in tiers], [59.0])
        self.assertEqual(tiers[0]["covered"], 62.0)

    def test_unrated_models_add_no_tiers(self):
        tiers = assign.schedule(["astra", "ghost"], {"astra": ASTRA})
        self.assertEqual([tier["target"] for tier in tiers], [45.65, 57.05, 60.29])
        for tier in tiers:
            self.assertEqual([model for model, _, _ in tier["entries"]], ["astra"])


if __name__ == "__main__":
    unittest.main()
