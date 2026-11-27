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
