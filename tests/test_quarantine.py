import unittest

from auto_router_pool import quarantine


class FakeClient:
    def __init__(self):
        self.calls = []

    def post(self, path, body, timeout=None):
        self.calls.append(("POST", path, body))
        return {}

    def patch(self, path, body):
        self.calls.append(("PATCH", path, body))
        return {}


def row(name, mid, blocked=False, marked=False, window=400000):
    info = {"id": mid, "mode": "chat", "blocked": blocked, "max_input_tokens": window}
    if marked:
        info[quarantine.MARKER] = True
    return {"model_name": name, "litellm_params": {"model": "openai/" + name},
            "model_info": info}


def budget_row(name, mid, credential="cred", blocked=False, marked=False, order=1,
               mode="chat"):
    info = {"id": mid, "mode": mode, "blocked": blocked}
    if marked:
        info[quarantine.BUDGET_MARKER] = True
    return {
        "model_name": name,
        "litellm_params": {"litellm_credential_name": credential, "order": order},
        "model_info": info,
    }


class BudgetSweepTest(unittest.TestCase):
    def test_budget_exceeded_blocks_every_deployment_sharing_the_credential(self):
        rows = [budget_row("a", "1"), budget_row("b", "2", order=2)]
        client = FakeClient()
        actions = quarantine.budget_sweep(
            client, rows, lambda mid: "budget_exceeded", apply=True
        )
        self.assertEqual([item["action"] for item in actions], ["blocked", "blocked"])
        self.assertTrue(all(
            row["model_info"]["blocked"]
            and row["model_info"][quarantine.BUDGET_MARKER]
            for row in rows
        ))
        self.assertEqual(
            [call[1] for call in client.calls if call[0] == "POST"],
            ["/model/block", "/model/block"],
        )
        self.assertEqual(
            [call[1] for call in client.calls if call[0] == "PATCH"],
            ["/model/1/update", "/model/2/update"],
        )

    def test_an_already_marked_deployment_is_left_alone(self):
        rows = [budget_row("a", "1", blocked=True, marked=True)]
        client = FakeClient()
        self.assertEqual(
            quarantine.budget_sweep(
                client, rows, lambda mid: "budget_exceeded", apply=True
            ),
            [],
        )
        self.assertEqual(client.calls, [])

    def test_an_already_blocked_deployment_is_adopted_without_blocking_again(self):
        rows = [budget_row("a", "1", blocked=True)]
        client = FakeClient()
        actions = quarantine.budget_sweep(
            client, rows, lambda mid: "budget_exceeded", apply=True
        )
        self.assertEqual(actions[0]["action"], "adopted")
        self.assertNotIn("/model/block", [call[1] for call in client.calls])
        self.assertIn(
            ("PATCH", "/model/1/update", {"model_info": {quarantine.BUDGET_MARKER: True}}),
            client.calls,
        )

    def test_recovery_unblocks_only_our_budget_marks(self):
        rows = [
            budget_row("a", "1", blocked=True, marked=True),
            budget_row("b", "2", blocked=True),
        ]
        client = FakeClient()
        actions = quarantine.budget_sweep(client, rows, lambda mid: "ok", apply=True)
        self.assertEqual(actions, [{
            "model": "a", "deployment": "1",
            "action": "unblocked", "reason": "budget_recovered",
        }])
        self.assertFalse(rows[0]["model_info"]["blocked"])
        self.assertFalse(rows[0]["model_info"][quarantine.BUDGET_MARKER])
        self.assertTrue(rows[1]["model_info"]["blocked"])
        self.assertEqual(
            [call[2] for call in client.calls if call[1] == "/model/unblock"],
            [{"model_id": "1"}],
        )

    def test_another_failure_changes_nothing(self):
        rows = [budget_row("a", "1")]
        self.assertEqual(
            quarantine.budget_sweep(
                FakeClient(), rows, lambda mid: "unhealthy", apply=True
            ),
            [],
        )
        self.assertFalse(rows[0]["model_info"]["blocked"])

    def test_a_preview_writes_nothing(self):
        rows = [budget_row("a", "1")]
        client = FakeClient()
        actions = quarantine.budget_sweep(client, rows, lambda mid: "budget_exceeded")
        self.assertEqual(actions[0]["action"], "blocked")
        self.assertEqual(client.calls, [])


class SweepTest(unittest.TestCase):
    def test_a_failing_group_is_narrowed_not_removed(self):
        rows = [row("g", "a"), row("g", "b")]
        client = FakeClient()
        actions = quarantine.sweep(client, rows, ["g"], lambda name: (False, "unavailable_429"), apply=True)
        self.assertEqual(actions, [{"model": "g", "deployment": "b", "action": "blocked",
                                    "reason": "unavailable_429"}])
        self.assertFalse(rows[0]["model_info"]["blocked"])
        self.assertTrue(rows[1]["model_info"]["blocked"])
        self.assertTrue(rows[1]["model_info"][quarantine.MARKER])

    def test_the_last_live_deployment_is_left_alone(self):
        rows = [row("g", "a")]
        actions = quarantine.sweep(FakeClient(), rows, ["g"], lambda name: (False, "unavailable_500"))
        self.assertEqual(actions[0]["action"], "kept")
        self.assertFalse(rows[0]["model_info"]["blocked"])

    def test_only_our_own_block_is_lifted(self):
        rows = [row("g", "a", blocked=True, marked=True), row("g", "b", blocked=True)]
        client = FakeClient()
        actions = quarantine.sweep(client, rows, ["g"], lambda name: (True, "ok"), apply=True)
        self.assertEqual(actions, [{"model": "g", "deployment": "a", "action": "unblocked"}])
        self.assertIn(("POST", "/model/unblock", {"model_id": "a"}), client.calls)
        self.assertNotIn(("POST", "/model/unblock", {"model_id": "b"}), client.calls)

    def test_a_preview_run_writes_nothing(self):
        rows = [row("g", "a"), row("g", "b")]
        client = FakeClient()
        quarantine.sweep(client, rows, ["g"], lambda name: (False, "unavailable_429"))
        self.assertEqual(client.calls, [])
