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

class LiveFailureSweepTest(unittest.TestCase):
    def test_blocks_managed_deployment_failing_on_live_surface_even_when_acompletion_passes(self):
        from datetime import datetime, timezone

        rows = [
            {
                "model_name": "glm-5.3-flash",
                "litellm_params": {"model": "hosted_vllm/z-ai/glm-5.3-flash"},
                "model_info": {"id": "glm-1", "mode": "chat", "blocked": False, "max_input_tokens": 1000000},
            },
            row("claude-opus-5-5", "opus-1"),
        ]
        stats = [
            {"model_id": "glm-1", "model": "hosted_vllm/z-ai/glm-5.3-flash", "call_type": "acompletion", "ok": 40, "bad": 0},
            {"model_id": "glm-1", "model": "hosted_vllm/z-ai/glm-5.3-flash", "call_type": "anthropic_messages", "ok": 1, "bad": 41},
            {"model_id": "opus-1", "model": "openai/claude-opus-5-5", "call_type": "anthropic_messages", "ok": 300, "bad": 2},
        ]
        client = FakeClient()
        now = datetime(2026, 11, 27, 10, 0, 0, tzinfo=timezone.utc)
        actions = quarantine.live_failure_sweep(
            client, rows, ["glm-5.3-flash", "claude-opus-5-5"], stats, apply=True, now=now
        )
        self.assertEqual(actions, [{
            "model": "glm-5.3-flash",
            "deployment": "glm-1",
            "action": "blocked",
            "reason": "anthropic_messages:41/42",
        }])
        self.assertTrue(rows[0]["model_info"]["blocked"])
        self.assertTrue(rows[0]["model_info"][quarantine.LIVE_FAILURE_MARKER])
        self.assertEqual(rows[0]["model_info"][quarantine.LIVE_FAILURE_AT], "2026-11-27T10:00:00Z")
        self.assertFalse(rows[1]["model_info"]["blocked"])
        self.assertIn(("POST", "/model/block", {"model_id": "glm-1"}), client.calls)

    def test_synthetic_sweep_does_not_unblock_live_failure_marker(self):
        rows = [
            {
                "model_name": "glm-5.3-flash",
                "litellm_params": {"model": "hosted_vllm/z-ai/glm-5.3-flash"},
                "model_info": {
                    "id": "glm-1",
                    "mode": "chat",
                    "blocked": True,
                    "max_input_tokens": 1000000,
                    quarantine.LIVE_FAILURE_MARKER: True,
                },
            }
        ]
        client = FakeClient()
        actions = quarantine.sweep(client, rows, ["glm-5.3-flash"], lambda name: (True, "ok"), apply=True)
        self.assertEqual(actions, [])
        self.assertTrue(rows[0]["model_info"]["blocked"])
        self.assertEqual(client.calls, [])

    def test_transport_migration_resets_stale_model_string_failures(self):
        rows = [
            {
                "model_name": "nemotron-3.5-lightning",
                "litellm_params": {"model": "hosted_vllm/nvidia/nemotron-3.5-lightning"},
                "model_info": {"id": "nem-1", "mode": "chat", "blocked": False, "max_input_tokens": 1000000},
            },
            row("claude-opus-5-5", "opus-1"),
        ]
        stats = [
            {"model_id": "nem-1", "model": "openai/nvidia/nemotron-3.5-lightning", "call_type": "aresponses", "ok": 0, "bad": 10},
        ]
        client = FakeClient()
        actions = quarantine.live_failure_sweep(client, rows, ["nemotron-3.5-lightning"], stats, apply=True)
        self.assertEqual(actions, [])
        self.assertFalse(rows[0]["model_info"]["blocked"])

    def test_unmanaged_models_are_ignored(self):
        rows = [
            {
                "model_name": "clef",
                "litellm_params": {"model": "hosted_vllm/clef"},
                "model_info": {"id": "clef-1", "mode": "chat", "blocked": False, "max_input_tokens": 200000},
            },
            row("claude-opus-5-5", "opus-1"),
        ]
        stats = [
            {"model_id": "clef-1", "model": "hosted_vllm/clef", "call_type": "acompletion", "ok": 0, "bad": 10},
        ]
        actions = quarantine.live_failure_sweep(FakeClient(), rows, ["claude-opus-5-5"], stats, apply=True)
        self.assertEqual(actions, [])
        self.assertFalse(rows[0]["model_info"]["blocked"])

    def test_never_empties_last_eligible_chat_model(self):
        rows = [
            {
                "model_name": "glm-5.3-flash",
                "litellm_params": {"model": "hosted_vllm/z-ai/glm-5.3-flash"},
                "model_info": {"id": "glm-1", "mode": "chat", "blocked": False, "max_input_tokens": 1000000},
            }
        ]
        stats = [
            {"model_id": "glm-1", "model": "hosted_vllm/z-ai/glm-5.3-flash", "call_type": "anthropic_messages", "ok": 0, "bad": 10},
        ]
        client = FakeClient()
        actions = quarantine.live_failure_sweep(client, rows, ["glm-5.3-flash"], stats, apply=True)
        self.assertEqual(actions[0]["action"], "kept")
        self.assertFalse(rows[0]["model_info"]["blocked"])
        self.assertEqual(client.calls, [])

    def test_cooldown_unblocks_only_our_live_failure_mark_after_retry_hours(self):
        from datetime import datetime, timezone

        rows = [
            {
                "model_name": "glm-5.3-flash",
                "litellm_params": {"model": "hosted_vllm/z-ai/glm-5.3-flash"},
                "model_info": {
                    "id": "glm-auto",
                    "mode": "chat",
                    "blocked": True,
                    "max_input_tokens": 1000000,
                    quarantine.LIVE_FAILURE_MARKER: True,
                    quarantine.LIVE_FAILURE_AT: "2026-11-26T09:00:00Z",
                    quarantine.LIVE_FAILURE_REASON: "anthropic_messages:41/42",
                },
            },
            {
                "model_name": "glm-5.3-flash",
                "litellm_params": {"model": "hosted_vllm/z-ai/glm-5.3-flash"},
                "model_info": {
                    "id": "glm-manual",
                    "mode": "chat",
                    "blocked": True,
                    "max_input_tokens": 1000000,
                },
            },
        ]
        client = FakeClient()
        early = datetime(2026, 11, 27, 8, 59, 0, tzinfo=timezone.utc)
        self.assertEqual(quarantine.live_failure_sweep(client, rows, [], [], apply=True, now=early), [])
        later = datetime(2026, 11, 27, 9, 0, 1, tzinfo=timezone.utc)
        actions = quarantine.live_failure_sweep(client, rows, [], [], apply=True, now=later)
        self.assertEqual(actions, [{
            "model": "glm-5.3-flash",
            "deployment": "glm-auto",
            "action": "unblocked",
            "reason": "live_failure_cooldown_elapsed",
        }])
        self.assertFalse(rows[0]["model_info"]["blocked"])
        self.assertFalse(rows[0]["model_info"][quarantine.LIVE_FAILURE_MARKER])
        self.assertTrue(rows[1]["model_info"]["blocked"])

    def test_preview_writes_nothing(self):
        rows = [
            {
                "model_name": "glm-5.3-flash",
                "litellm_params": {"model": "hosted_vllm/z-ai/glm-5.3-flash"},
                "model_info": {"id": "glm-1", "mode": "chat", "blocked": False, "max_input_tokens": 1000000},
            },
            row("claude-opus-5-5", "opus-1"),
        ]
        stats = [
            {"model_id": "glm-1", "model": "hosted_vllm/z-ai/glm-5.3-flash", "call_type": "anthropic_messages", "ok": 0, "bad": 8},
        ]
        client = FakeClient()
        actions = quarantine.live_failure_sweep(client, rows, ["glm-5.3-flash"], stats, apply=False)
        self.assertEqual(actions[0]["action"], "blocked")
        self.assertEqual(client.calls, [])

    def test_spend_failure_stats_fails_closed_on_runner_error(self):
        class Fail:
            returncode = 1
            stdout = ""

        self.assertEqual(quarantine.spend_failure_stats(runner=lambda *a, **k: Fail()), [])
