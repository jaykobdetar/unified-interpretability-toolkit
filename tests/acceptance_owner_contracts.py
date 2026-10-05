"""Pure normal-owner client controls with fake transport/clock; no I/O/children."""

import unittest
from types import SimpleNamespace
from acceptance.owner_client import OwnerClient, clean, parity
from acceptance.render_contract import assert_render_rules


class OwnerTests(unittest.TestCase):
    def test_combined_renderer_rule_contract(self):
        ids = [
            "global_linear",
            "global_asinh",
            "tensor_linear",
            "tensor_asinh",
            "tensor_magnitude",
            "tensor_magnitude_asinh",
            "tensor_robust99",
            "tensor_signed_percentile",
        ]
        assert_render_rules({"rules": [{"id": rule} for rule in ids]})
        faults = [
            ids[:-1],
            ids + [ids[0]],
            [ids[0], *ids[2:], ids[0]],
            ["unknown", *ids[1:]],
            list(reversed(ids)),
        ]
        for fault in faults:
            with self.subTest(ids=fault), self.assertRaises(AssertionError):
                assert_render_rules({"rules": [{"id": rule} for rule in fault]})

    def client(self, responses):
        clock = SimpleNamespace(now=0.0)
        calls = []

        def transport(action, data):
            calls.append((action, data))
            return responses.pop(0)

        return (
            OwnerClient(
                "http://127.0.0.1:8816",
                clock=lambda: clock.now,
                sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
                transport=transport,
            ),
            calls,
        )

    def test_exited_pending_owner_blocks_replacement_until_terminal(self):
        responses = [
            {
                "session": "one",
                "status": "running",
                "worker_alive": True,
                "steps": [],
                "details": {},
            },
            {
                "session": "one",
                "status": "stopping",
                "worker_alive": False,
                "steps": [],
                "details": {"cleanup_pending": True},
            },
            {
                "session": "one",
                "status": "cancelled",
                "worker_alive": False,
                "steps": [],
                "details": {},
            },
            {
                "session": "two",
                "status": "loading",
                "worker_alive": True,
                "steps": [],
                "details": {},
            },
        ]
        client, calls = self.client(responses)
        client.start({"mode": "sweep"})
        client.cancel()
        self.assertFalse(clean(client.snapshot))
        with self.assertRaises(RuntimeError):
            client.start({"edits": []})
        self.assertEqual(len(calls), 2)
        client.finish()
        client.start({"edits": []})
        self.assertEqual(client.session, "two")
        self.assertEqual(calls[1][1], {"session": "one"})
        self.assertEqual(calls[2][1], {"session": "one"})
        self.assertEqual(client.starts, 2)

    def test_total_deadline_refuses_more_work(self):
        client, calls = self.client([])
        client.deadline = 0
        with self.assertRaises(TimeoutError):
            client.start({"edits": []})
        self.assertEqual(calls, [])

    def test_raced_completion_not_labeled_cancellation(self):
        client, _ = self.client(
            [
                {
                    "session": "one",
                    "status": "loading",
                    "worker_alive": True,
                    "steps": [],
                    "details": {},
                },
                {
                    "session": "one",
                    "status": "complete",
                    "worker_alive": False,
                    "steps": [],
                    "details": {},
                },
            ]
        )
        client.start({})
        client.cancel()
        self.assertEqual(client.finish()["status"], "complete")
        self.assertTrue(client.cleanup())

    def test_uncertain_admission_never_claims_cleanup(self):
        client, _ = self.client([])

        def lost(*_args):
            raise TimeoutError("fake lost admission response")

        client.transport = lost
        with self.assertRaises(TimeoutError):
            client.start({})
        self.assertFalse(client.cleanup())

    def test_control_parity_is_exact_and_excludes_timing(self):
        snap = {
            "session": "one",
            "status": "complete",
            "worker_alive": False,
            "details": {
                "baseline": {"generated_ids": [1], "generated_text": "a"},
                "edited": {"generated_ids": [1], "generated_text": "a"},
            },
            "steps": [
                {
                    "alignment": "matched_prefix",
                    "activation": [0] * 576,
                    "candidates": [
                        {"baseline_logit": 1, "edited_logit": 1, "delta": 0}
                    ],
                    "compute_ms": 123,
                }
            ],
        }
        first = parity(snap)
        snap["session"] = "two"
        snap["steps"][0]["compute_ms"] = 456
        self.assertEqual(first, parity(snap))
        snap["steps"][0]["candidates"][0]["delta"] = 1
        with self.assertRaises(AssertionError):
            parity(snap)


if __name__ == "__main__":
    unittest.main()
