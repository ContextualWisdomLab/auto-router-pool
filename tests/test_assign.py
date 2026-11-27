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

    def test_frontier_keeps_peak_efforts_without_ladders(self):
        routers = tiers.routers("sol", "astra", "budget", "old", (), {"astra": "max"})
        frontier = routers["auto"]["최고난도 추론·심층 연구·복잡한 장기 작업"]
        self.assertEqual(frontier, [("astra", "max"), ("sol", "xhigh"), ("old", "xhigh")])


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
