import json
import tempfile
import unittest
from pathlib import Path

from auto_router_pool import registry

GUEST = {
    "model_name": "guest",
    "litellm_params": {"model": "openrouter/x/y", "api_base": "https://example/v1"},
    "model_info": {"mode": "chat", "is_free": True},
}


class FakeClient:
    def __init__(self, create=None, answers=True, decision=None):
        self.calls = []
        self.create = create or {"model_id": "mid-1"}
        self.answers = answers
        self.decision = decision or {"error_status": 400}

    def post(self, path, body, timeout=None):
        self.calls.append((path, body))
        if path == "/model/new":
            return self.create
        if path == "/model/delete":
            return {"success": True}
        if path == "/v1/chat/completions":
            return {"model": body["model"]} if self.answers else {"error_status": 401}
        if path in ("/v1/decisions", "/v1/systemone"):
            return self.decision
        return {}

    def paths(self):
        return [path for path, _ in self.calls]


class LoadTest(unittest.TestCase):
    def write(self, payload):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        handle.write(json.dumps(payload))
        handle.close()
        return Path(handle.name)

    def test_a_missing_file_declares_nothing(self):
        self.assertEqual(registry.load_guests(Path("/nonexistent/guests.json")), [])

    def test_entries_may_be_wrapped_or_bare(self):
        self.assertEqual(len(registry.load_guests(self.write({"models": [GUEST]}))), 1)
        self.assertEqual(len(registry.load_guests(self.write([GUEST]))), 1)

    def test_a_declaration_without_an_upstream_is_an_error(self):
        broken = {"model_name": "guest", "litellm_params": {}}
        with self.assertRaises(ValueError):
            registry.load_guests(self.write([broken]))


class EnsureTest(unittest.TestCase):
    def test_a_missing_guest_is_registered_and_kept_when_it_answers(self):
        client = FakeClient()
        results = registry.ensure_guests(client, [GUEST], [])
        self.assertEqual(results, [{"model": "guest", "action": "registered"}])
        self.assertEqual(client.paths(), ["/model/new", "/v1/chat/completions"])

    def test_a_guest_that_cannot_answer_is_deleted_again(self):
        client = FakeClient(answers=False)
        results = registry.ensure_guests(client, [GUEST], [])
        self.assertEqual(results[0]["action"], "reverted")
        self.assertEqual(results[0]["detail"], "unavailable_401")
        self.assertEqual(client.paths()[-1], "/model/delete")

    def test_an_already_registered_guest_is_left_alone(self):
        client = FakeClient()
        rows = [{"model_name": "guest", "model_info": {}}]
        self.assertEqual(registry.ensure_guests(client, [GUEST], rows),
                         [{"model": "guest", "action": "present"}])
        self.assertEqual(client.calls, [])

    def test_a_decision_guest_is_probed_on_its_endpoint_not_chat(self):
        guest = dict(GUEST, model_info={"mode": "decision"})
        client = FakeClient(decision={"error_status": 400})
        results = registry.ensure_guests(client, [guest], [])
        self.assertEqual(results, [{"model": "guest", "action": "registered"}])
        self.assertIn("/v1/decisions", client.paths())
        self.assertNotIn("/v1/chat/completions", client.paths())

    def test_a_decision_guest_whose_alias_is_wrong_is_deleted(self):
        guest = dict(GUEST, model_info={"mode": "decision"})
        client = FakeClient(decision={"error_status": 403})
        results = registry.ensure_guests(client, [guest], [])
        self.assertEqual(results[0]["action"], "reverted")
        self.assertEqual(client.paths()[-1], "/model/delete")

    def test_a_refused_registration_is_reported_not_retried(self):
        client = FakeClient(create={"error_status": 400})
        results = registry.ensure_guests(client, [GUEST], [])
        self.assertEqual(results, [{"model": "guest", "action": "failed",
                                    "detail": "create_400"}])
        self.assertEqual(client.paths(), ["/model/new"])


if __name__ == "__main__":
    unittest.main()
