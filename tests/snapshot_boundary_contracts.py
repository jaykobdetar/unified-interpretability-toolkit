"""Snapshot arithmetic and ownership contracts using the existing in-memory OS."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from atlas_host import profile_snapshot as snapshot
import profile_worker_primitives as primitive


class SnapshotBoundaries(unittest.TestCase):
    def success(self, operation):
        try:
            return operation()
        except Exception as error:
            self.fail(f"Expected successful in-memory snapshot operation: {error!r}")

    def refuses(self, operation, message):
        try:
            operation()
        except ValueError as error:
            self.assertEqual(str(error), message)
        except Exception as error:
            self.fail(f"Expected exact ValueError, received {error!r}")
        else:
            self.fail("Expected exact ValueError")

    def fixture(self):
        case = primitive.Base("runTest")
        case.setUp()
        self.addCleanup(case.doCleanups)
        return case

    def store(self):
        case = self.fixture()
        store = self.success(lambda: snapshot.SnapshotStore(primitive.selected(), 17))
        return case, store

    def published(self, visited=5):
        case, store = self.store()
        revision = self.success(lambda: case.candidate(store, visited))
        info = self.success(lambda: case.publish(store, revision))
        self.assertIsInstance(info, snapshot.SnapshotInfo)
        self.assertEqual(info.visited, visited)
        return case, store, revision

    def test_layout_and_identity_are_exact(self):
        selected = primitive.selected()
        raw = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
        frame = 168 + len(raw) + 48 * 7
        self.assertEqual(
            self.success(lambda: snapshot.layout(selected)),
            (frame, 56 * 7 + 2 * frame + 2228224, raw),
        )
        expected = hashlib.sha256(
            json.dumps(
                ["weight-atlas-strength-v1", selected, 17, "swap-or-not-8-v1"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).digest()
        self.assertEqual(
            self.success(lambda: snapshot.profile_identity(selected, 17)), expected
        )

    def test_mixer_and_destination_match_independent_small_reference(self):
        # Fixed SplitMix64 finalizer vectors, separate from production calls.
        for value, expected in (
            (0, 0),
            (1, 6238072747940578789),
            (17, 3471015484745077182),
            (18446744073709551615, 13029008266876403067),
        ):
            self.assertEqual(self.success(lambda: snapshot._mix(value)), expected)
        for total in (2, 7, 12, 35):
            for seed in (0, 17, 4294967295):
                expected = primitive.reference_permutation(total, seed)
                actual = self.success(
                    lambda: [snapshot.destination(i, total, seed) for i in range(total)]
                )
                self.assertEqual(actual, expected)

    def test_deadline_is_exclusive_and_samples_once(self):
        clock = Mock(return_value=4.5)
        self.assertIsNone(self.success(lambda: snapshot._deadline(5, clock)))
        clock.assert_called_once_with()
        self.refuses(
            lambda: snapshot._deadline(5, lambda: 5),
            "Snapshot deadline exhausted; Restart may be required",
        )

    def test_memfd_creation_and_seal_call_contracts_are_inert(self):
        create = Mock(return_value=41)
        fake_os = SimpleNamespace(
            memfd_create=create, MFD_CLOEXEC=1, MFD_ALLOW_SEALING=2
        )
        call = Mock(return_value=15)
        fake_fcntl = SimpleNamespace(F_ADD_SEALS=1033, fcntl=call)
        with (
            patch.object(snapshot, "os", fake_os),
            patch.object(snapshot, "fcntl", fake_fcntl),
        ):
            self.assertEqual(self.success(snapshot.new_memfd), 41)
            create.assert_called_once_with("atlas-profile", 3)
            self.assertIsNone(self.success(lambda: snapshot.seal(41)))
            call.assert_called_once_with(41, 1033, snapshot.SEALS)
        with patch.object(snapshot, "os", SimpleNamespace()):
            self.refuses(
                snapshot.new_memfd, "Sealed memfd is unavailable; no disk fallback"
            )

    def test_exact_storage_metadata_and_positioned_reads(self):
        case = self.fixture()
        fd = case.os.put(b"abcdefgh")
        self.assertIsNone(self.success(lambda: snapshot.check_sealed(fd, 8)))
        self.refuses(
            lambda: snapshot.check_sealed(fd, 9),
            "Snapshot is not exact-sized anonymous storage",
        )
        self.assertEqual(self.success(lambda: snapshot._read(fd, 3, 2)), b"cde")
        self.refuses(lambda: snapshot._read(fd, 9, 0), "Truncated snapshot")

    def test_streaming_and_drained_validation_return_all_exact_fields(self):
        case = self.fixture()
        selected = primitive.selected()
        raw, *_ = primitive.frame(selected, 17, 5)
        revision = hashlib.sha256(raw).hexdigest()
        expected = snapshot.SnapshotInfo(
            revision, 5, 12, len(raw), 56 * 7 + 2 * len(raw) + 2228224
        )
        for streaming in (True, False):
            with self.subTest(streaming=streaming):
                fd = case.os.put(raw)
                if streaming:

                    def drain():
                        steps = snapshot.validation_steps(
                            fd, selected, 17, revision, deadline=5, clock=lambda: 0
                        )
                        yielded = []
                        while True:
                            try:
                                yielded.append(next(steps))
                            except StopIteration as done:
                                return yielded, done.value

                    yielded, info = self.success(drain)
                    self.assertTrue(yielded)
                    self.assertTrue(all(value is None for value in yielded))
                else:
                    info = self.success(
                        lambda: snapshot.validate(
                            fd, selected, 17, revision, deadline=5, clock=lambda: 0
                        )
                    )
                self.assertEqual(info, expected)

    def test_constructor_charges_and_candidate_abort(self):
        case = self.fixture()
        selected = primitive.selected()
        diagnostics = object()
        store = self.success(
            lambda: snapshot.SnapshotStore(selected, 17, diagnostics=diagnostics)
        )
        self.assertIs(store.diagnostics, diagnostics)
        self.assertEqual(store.selected, selected)
        self.assertIsNot(store.selected, selected)
        self.assertEqual(
            (
                store.seed,
                store.latest,
                store.pending,
                store.info,
                store.readers,
                store.publication_serial,
                store._owned,
                store._uncertain,
            ),
            (17, None, None, None, 0, 0, (), frozenset()),
        )
        self.assertEqual(store.owned_storage_bytes, 0)
        self.assertIsNone(self.success(store.require_settled))
        fd = self.success(store.begin)
        self.assertEqual((fd, store.pending, store._owned), (21, 21, (21,)))
        self.assertEqual(store.owned_storage_bytes, store.frame_bytes)
        self.refuses(store.begin, "Snapshot candidate/read already exists")
        self.assertIsNone(self.success(store.abort_candidate))
        self.assertIsNone(store.pending)
        self.assertEqual((store._owned, case.os.closed, case.os.files), ((), [21], {}))
        self.assertIsNone(self.success(store.abort_candidate))
        self.assertEqual(case.os.closed, [21])

    def test_close_owned_and_settlement_bookkeeping(self):
        case, store = self.store()
        fd = self.success(store.begin)
        self.assertIsNone(self.success(lambda: store._close_owned(fd)))
        self.assertEqual((store._owned, case.os.closed, case.os.files), ((), [fd], {}))
        self.refuses(
            lambda: store._close_owned(fd),
            "Snapshot close disposition uncertain; do not retry descriptor",
        )
        self.refuses(
            store.require_settled, "Uncertain retiring snapshot storage still owned"
        )
        store.pending = None
        self.assertIsNone(self.success(store.require_settled))

    def test_publication_steps_and_publish_preserve_records_and_order(self):
        for streaming in (True, False):
            with self.subTest(streaming=streaming):
                case, store = self.store()
                revision = self.success(lambda: case.candidate(store, 5))
                fd = store.pending
                calls = []
                kwargs = dict(
                    minimum_visited=0,
                    maximum_visited=12,
                    deadline=5,
                    source_check=lambda: calls.append("source"),
                    final_check=lambda: calls.append("final"),
                    clock=lambda: 0,
                )
                if streaming:

                    def drain():
                        steps = store.publication_steps(revision, **kwargs)
                        while True:
                            try:
                                self.assertIsNone(next(steps))
                            except StopIteration as done:
                                return done.value

                    info = self.success(drain)
                else:
                    info = self.success(lambda: store.publish(revision, **kwargs))
                expected = snapshot.SnapshotInfo(
                    revision, 5, 12, store.frame_bytes, store.live_bytes
                )
                self.assertEqual(info, expected)
                self.assertEqual(
                    (store.latest, store.pending, store.info, store.publication_serial),
                    (fd, None, expected, 1),
                )
                self.assertEqual(calls, ["source", "source", "final"])
                self.assertEqual(store._owned, (fd,))

    def test_page_handle_pins_and_releases_one_reader(self):
        _case, store, revision = self.published()

        def read():
            with store.page_handle(revision, expected_publication=1) as held:
                self.assertEqual(store.readers, 1)
                self.assertEqual(held, (store.latest, store.info))
            self.assertEqual(store.readers, 0)

        self.success(read)

    def test_page_values_and_binding_copy_match_independent_records(self):
        _case, store, revision = self.published()
        _raw, sums, _values, _perm = primitive.frame(primitive.selected(), 17, 5)
        calls = []
        page = self.success(
            lambda: store.page(
                revision,
                "rows",
                0,
                3,
                source_check=lambda: calls.append("source"),
                expected_publication=1,
            )
        )

        def records(items):
            return [
                dict(
                    index=i,
                    sum_abs=value,
                    visited_count=count,
                    expected_count=4,
                    mean_abs=None if count == 0 else value / count,
                    complete=count == 4,
                )
                for i, (value, _correction, count) in enumerate(items)
            ]

        self.assertEqual(
            page,
            dict(
                revision=revision,
                binding=primitive.selected(),
                axis="rows",
                start=0,
                end=3,
                axis_length=3,
                visited_values=5,
                total_values=12,
                original=records(sums[:3]),
                control=records(sums[7:10]),
            ),
        )
        self.assertEqual(calls, ["source", "source"])
        self.assertIsNot(page["binding"], store.selected)

    def test_close_retires_latest_and_pending_without_repeated_close(self):
        case, store, _revision = self.published()
        latest = store.latest
        pending = self.success(store.begin)
        self.assertEqual(store.owned_storage_bytes, 2 * store.frame_bytes)
        self.assertIsNone(self.success(store.close))
        self.assertEqual(
            (store.latest, store.pending, store.info, store._owned, case.os.files),
            (None, None, None, (), {}),
        )
        self.assertEqual(case.os.closed, [latest, pending])
        self.assertEqual(store.owned_storage_bytes, 0)
        self.assertIsNone(self.success(store.close))
        self.assertEqual(case.os.closed, [latest, pending])


if __name__ == "__main__":
    unittest.main()
