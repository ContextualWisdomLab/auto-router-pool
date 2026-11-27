import unittest

from auto_router_pool import live


def row(name, blocked=False, window=500000):
    return {"model_name": name, "model_info": {"blocked": blocked, "max_input_tokens": window}}


def router(solvers):
    return {
        "model_name": "auto",
        "model_info": {},
        "litellm_params": {"complexity_router_config": {"tiers": {"SIMPLE": list(solvers)}}},
    }


PRICES = {
    "gpt-6-luna": {"input_cost_per_token": 1e-7, "output_cost_per_token": 5e-7},
    "gemini-3.8-flash": {"input_cost_per_token": 7.5e-7, "output_cost_per_token": 3.75e-6},
    "claude-sonnet-5-5": {"input_cost_per_token": 2e-6, "output_cost_per_token": 1e-5},
    "grok-4.6": {"input_cost_per_token": 1e-6, "output_cost_per_token": 1e-5},
}


class FamilyTest(unittest.TestCase):
    def test_prefixes(self):
        self.assertEqual(live.family("gpt-6-luna"), "openai")
        self.assertEqual(live.family("claude-opus-5-5"), "anthropic")
        self.assertEqual(live.family("gemini-3.8-flash"), "gemini")
        self.assertEqual(live.family("grok-4.7"), "xai")
        self.assertIsNone(live.family("laguna-xs-2.1"))


class ChooseTerminalTest(unittest.TestCase):
    def test_prefers_a_family_the_tiers_do_not_use(self):
        rows = [row("gpt-6-luna"), row("gpt-6-luna"), row("gemini-3.8-flash"),
                row("gemini-3.8-flash"), row("gpt-6.2-sol")]
        chosen, _ = live.choose_terminal(rows, PRICES, {"gpt-6.2-sol"})
        self.assertEqual(chosen, "gemini-3.8-flash")

    def test_cheapest_inside_the_preferred_family(self):
        rows = [row("gemini-3.8-flash"), row("gemini-3.8-flash"),
                row("claude-sonnet-5-5"), row("claude-sonnet-5-5")]
        chosen, _ = live.choose_terminal(rows, PRICES, set())
        self.assertEqual(chosen, "gemini-3.8-flash")

    def test_window_below_the_pool_floor_is_rejected(self):
        rows = [row("gemini-3.8-flash", window=65536), row("gemini-3.8-flash", window=65536)]
        self.assertEqual(live.choose_terminal(rows, PRICES, set()), (None, {}))

    def test_a_solver_is_never_the_terminal(self):
        rows = [row("gpt-6-luna"), row("gpt-6-luna")]
        self.assertEqual(live.choose_terminal(rows, PRICES, {"gpt-6-luna"}), (None, {}))

    def test_internal_and_router_groups_are_excluded(self):
        rows = [row("__advisor_fable"), row("__advisor_fable"),
                row("auto-classify"), row("auto-classify")]
        self.assertEqual(live.choose_terminal(rows, PRICES, set()), (None, {}))

    def test_incumbent_is_kept_while_it_qualifies(self):
        rows = [row("gemini-3.8-flash"), row("gemini-3.8-flash"),
                row("claude-sonnet-5-5"), row("claude-sonnet-5-5")]
        chosen, _ = live.choose_terminal(rows, PRICES, set(), current="claude-sonnet-5-5")
        self.assertEqual(chosen, "claude-sonnet-5-5")

    def test_dead_incumbent_is_replaced(self):
        rows = [row("gemini-3.8-flash"), row("gemini-3.8-flash"),
                row("claude-sonnet-5-5", blocked=True), row("claude-sonnet-5-5", blocked=True)]
        chosen, _ = live.choose_terminal(rows, PRICES, set(), current="claude-sonnet-5-5")
        self.assertEqual(chosen, "gemini-3.8-flash")

    def test_facts_report_the_justification(self):
        rows = [row("gemini-3.8-flash"), row("gemini-3.8-flash", window=1048576),
                row("gemini-3.8-flash", blocked=True)]
        chosen, fact = live.choose_terminal(rows, PRICES, set())
        self.assertEqual(chosen, "gemini-3.8-flash")
        self.assertEqual(fact["unblocked"], 2)
        self.assertEqual(fact["window"], 1048576)


class RequiredTest(unittest.TestCase):
    def test_every_required_chain_ends_on_the_terminal(self):
        for terminal in ("gemini-3.8-flash", "claude-sonnet-5-5"):
            chains = live.required(terminal)
            for source, targets in chains.items():
                self.assertIn(terminal, targets, source)
                self.assertNotEqual(source, terminal)


class SanitizeTest(unittest.TestCase):
    def rows(self, dead_terminal_blocked=True):
        return [
            router(["gpt-6.2-sol"]),
            row("gpt-6.2-sol"),
            row("gemini-3.8-flash"),
            row("claude-sonnet-5-5"),
            row("grok-4.7", blocked=dead_terminal_blocked),
        ]

    def test_strips_blocked_targets_and_ends_chains_on_the_resort(self):
        settings = {"fallbacks": [{"gpt-6.2-sol": ["grok-4.7"]},
                                  {"gemini-3.8-flash": ["grok-4.7"]}]}
        result = live.sanitize(self.rows(), settings, PRICES)
        self.assertEqual(result["terminal"], "gemini-3.8-flash")
        graph = live.edges_from(result["fallbacks"])
        self.assertEqual(graph["gpt-6.2-sol"], ["gemini-3.8-flash"])
        self.assertNotIn("gemini-3.8-flash", graph)

    def test_keeps_a_live_target_and_drops_the_dead_one(self):
        settings = {"fallbacks": [{"gpt-6.2-sol": ["claude-sonnet-5-5", "grok-4.7"]}]}
        result = live.sanitize(self.rows(), settings, PRICES)
        self.assertEqual(live.edges_from(result["fallbacks"])["gpt-6.2-sol"],
                         ["claude-sonnet-5-5"])

    def test_is_idempotent(self):
        settings = {"fallbacks": [{"gpt-6.2-sol": ["grok-4.7"]}]}
        first = live.sanitize(self.rows(), settings, PRICES)
        second = live.sanitize(self.rows(), {"fallbacks": first["fallbacks"]}, PRICES)
        self.assertEqual(second["changes"], [])
        self.assertEqual(first["fallbacks"], second["fallbacks"])

    def test_targets_below_the_floor_are_stripped(self):
        rows = self.rows() + [row("laguna-xs-2.1", window=65536)]
        settings = {"fallbacks": [{"gpt-6.2-sol": ["gemini-3.8-flash", "laguna-xs-2.1"]}]}
        result = live.sanitize(rows, settings, PRICES)
        self.assertEqual(live.edges_from(result["fallbacks"])["gpt-6.2-sol"],
                         ["gemini-3.8-flash"])

    def test_required_chain_is_exact(self):
        rows = self.rows() + [row("claude-haiku-5-5"), row("gpt-6-luna"), row("gpt-6.1-sol")]
        settings = {"fallbacks": [{"claude-haiku-5-5": ["stale-model", "gpt-6-luna"]}]}
        result = live.sanitize(rows, settings, PRICES)
        self.assertEqual(live.edges_from(result["fallbacks"])["claude-haiku-5-5"],
                         live.required("gemini-3.8-flash")["claude-haiku-5-5"])

    def test_fails_closed_without_a_qualified_resort(self):
        rows = [router(["gpt-6.2-sol"]), row("gpt-6.2-sol"), row("grok-4.7", blocked=True)]
        with self.assertRaises(ValueError):
            live.sanitize(rows, {"fallbacks": [{"gpt-6.2-sol": ["grok-4.7"]}]}, PRICES)


if __name__ == "__main__":
    unittest.main()
