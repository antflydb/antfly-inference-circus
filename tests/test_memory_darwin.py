from __future__ import annotations

import dataclasses
import errno
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import memory_darwin as memory
import common as C

P = memory.PAGE_BYTES


def region(*, offset=0, size=8 * P, resident=4 * P, dirty=0,
           address=0x100000000, dev=1, ino=2, path="", mode=memory.SM_SHARED,
           shadow=0, flags=0):
    return memory.Region(address, size, offset, resident, dirty, 0, resident,
                         mode, shadow, flags, dev, ino, path)


class ResidencyIntervalTests(unittest.TestCase):
    def test_disjoint_file_windows_add_but_exact_aliases_do_not(self):
        first = region(resident=8 * P)
        alias = dataclasses.replace(first, address=first.address + 32 * P)
        disjoint = region(offset=8 * P, resident=8 * P, address=first.address + 64 * P)
        self.assertEqual(memory._offset_bounds([first, alias]), (8 * P, 8 * P))
        self.assertEqual(memory._offset_bounds([first, alias, disjoint]), (16 * P, 16 * P))

    def test_partial_aliases_retain_unknown_page_intersection(self):
        first = region(resident=4 * P)
        alias = dataclasses.replace(first, address=first.address + 32 * P)
        self.assertEqual(memory._offset_bounds([first, alias]), (4 * P, 8 * P))
        overlapping = region(offset=4 * P, resident=6 * P, address=first.address + 64 * P)
        self.assertEqual(memory._offset_bounds([first, overlapping]), (6 * P, 10 * P))

    def test_nested_aliases_use_union_capacity_not_largest_window_size(self):
        outer = region(size=16 * P, resident=12 * P)
        inner = region(offset=4 * P, size=8 * P, resident=8 * P,
                       address=outer.address + 32 * P)
        self.assertEqual(memory._offset_bounds([outer, inner]), (12 * P, 16 * P))

    def test_shadowed_private_copies_are_not_deduplicated_as_file_pages(self):
        original = region(resident=8 * P)
        copied = region(resident=8 * P, address=original.address + 32 * P,
                        mode=memory.SM_COW, shadow=1)
        self.assertEqual(memory._offset_bounds([original, copied]), (8 * P, 16 * P))
        self.assertEqual(memory._offset_bounds([copied, dataclasses.replace(
            copied, address=copied.address + 32 * P)]), (8 * P, 16 * P))

    def test_different_file_identities_are_not_collapsed(self):
        first = region(resident=8 * P)
        other = dataclasses.replace(first, dev=2)
        self.assertEqual(memory._offset_bounds([first, other]), (16 * P, 16 * P))

    def test_clean_candidates_exclude_dirty_without_claiming_ledger_attribution(self):
        item = region(resident=8 * P, dirty=3 * P)
        self.assertEqual(memory._offset_bounds([item], clean=True), (5 * P, 5 * P))
        self.assertEqual(memory._offset_bounds([item]), (8 * P, 8 * P))

    def test_offsets_and_addresses_beyond_four_gib_remain_distinct(self):
        first = region(offset=1 << 33, resident=8 * P, address=1 << 36)
        second = dataclasses.replace(first, offset=(1 << 33) + 8 * P,
                                     address=(1 << 36) + 32 * P)
        self.assertEqual(memory._offset_bounds([first, second]), (16 * P, 16 * P))

    def test_corrupt_counts_and_overflow_fail_instead_of_clamping(self):
        good = region()
        invalid = [dataclasses.replace(good, dirty=good.resident + P),
                   dataclasses.replace(good, resident=good.size + P),
                   dataclasses.replace(good, resident=-P),
                   dataclasses.replace(good, size=0),
                   dataclasses.replace(good, offset=memory.UINT64_LIMIT - P),
                   dataclasses.replace(good, address=memory.UINT64_LIMIT - P),
                   dataclasses.replace(good, share_mode=99)]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(memory.CoverageError):
                memory._offset_bounds([item])

    def test_process_tree_shared_pages_do_not_sum_lower_bounds(self):
        rows = [{"coverage_valid": True, "fp": 2 * P,
                 "mapped_storage_resident_lower_bytes": 8 * P,
                 "mapped_storage_resident_upper_bytes": 8 * P}] * 2
        self.assertEqual(memory.combine_samples(rows), {
            "valid": True, "lower_bytes": 8 * P, "upper_bytes": 20 * P})
        self.assertFalse(memory.combine_samples([])["valid"])
        self.assertFalse(memory.combine_samples([{"coverage_valid": False}])["valid"])


class FakeBackend:
    def __init__(self, rows=(), *, failure=None, identity_change=False, ledger_failure=False):
        self.rows = list(rows)
        self.failure = failure
        self.identity_change = identity_change
        self.ledger_failure = ledger_failure
        self.ledger_calls = 0
        self.cursors = []
        self.controls = []

    def ledger(self, pid):
        self.ledger_calls += 1
        if self.ledger_failure:
            raise memory.CoverageError("rusage permission denied")
        return {"fp": 2 * P, "rss": 3 * P, "lifetime_max": 1000 * P,
                "identity": (pid + (self.ledger_calls > 1 and self.identity_change), "uuid")}

    def control(self, name, value=None):
        self.controls.append((name, value))
        return 0 if name.endswith("footprint") else 1

    def region(self, pid, cursor):
        self.cursors.append(cursor)
        index = len(self.cursors) - 1
        if index < len(self.rows):
            return self.rows[index]
        if self.failure:
            raise self.failure
        raise memory.RegionEnd()


class ReaderContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.file = self.root / "model.bin"
        self.file.write_bytes(b"model")
        stat = self.file.stat()
        self.item = region(dev=stat.st_dev & 0xFFFFFFFF, ino=stat.st_ino,
                           path=str(self.file), resident=8 * P)

    def reader(self, backend, **kwargs):
        with patch.object(memory, "_Libproc", return_value=backend):
            return memory.DarwinMemoryReader([self.file], max_seconds=10, **kwargs)

    def test_success_returns_concurrent_interval_not_independent_peak_addition(self):
        backend = FakeBackend([self.item])
        result = self.reader(backend).read(123)
        self.assertTrue(result["coverage_valid"])
        self.assertEqual(result["fp"], 2 * P)
        self.assertEqual(result["lifetime_max"], 1000 * P)
        self.assertEqual(result["corrected_fp_lower_bytes"], 8 * P)
        self.assertEqual(result["corrected_fp_upper_bytes"], 10 * P)
        self.assertIsNone(result["corrected_fp"])
        self.assertFalse(result["corrected_fp_valid"])
        self.assertEqual(backend.cursors, [0, self.item.address + self.item.size])

    def test_copied_allocation_without_model_mapping_has_exact_ledger_interval(self):
        anonymous = dataclasses.replace(self.item, ino=0, path="")
        result = self.reader(FakeBackend([anonymous])).read(123)
        self.assertTrue(result["corrected_fp_valid"])
        self.assertEqual(result["corrected_fp"], 2 * P)
        self.assertEqual(result["mapped_storage_resident_lower_bytes"], 0)

    def test_unlinked_regular_storage_counts_without_a_pathname_receipt(self):
        temporary = dataclasses.replace(self.item, ino=self.item.ino + 1, path="",
                                        nlink=0, file_mode=stat.S_IFREG | 0o600)
        result = self.reader(FakeBackend([temporary])).read(123)
        self.assertTrue(result["coverage_valid"])
        self.assertEqual(result["unlinked_storage_observed_resident_bytes"], 8 * P)
        self.assertEqual(result["corrected_fp_lower_bytes"], 8 * P)
        self.assertEqual(result["corrected_fp_upper_bytes"], 10 * P)

    def test_foreign_linked_and_nonregular_files_are_not_temporary_storage(self):
        for nlink, mode in ((1, stat.S_IFREG), (0, stat.S_IFCHR), (None, stat.S_IFREG)):
            with self.subTest(nlink=nlink, mode=mode):
                foreign = dataclasses.replace(self.item, ino=self.item.ino + 1, path="",
                                              nlink=nlink, file_mode=mode)
                result = self.reader(FakeBackend([foreign])).read(123)
                self.assertEqual(result["corrected_fp"], 2 * P)
                self.assertEqual(result["unlinked_storage_observed_resident_bytes"], 0)

    def test_permission_and_partial_errors_do_not_become_end_of_regions(self):
        for error in (memory.CoverageError(f"permission errno={errno.EPERM}"),
                      memory.CoverageError("short region read")):
            with self.subTest(error=error):
                result = self.reader(FakeBackend([self.item], failure=error)).read(123)
                self.assertFalse(result["coverage_valid"])
                self.assertFalse(result["top_level_walk_complete"])
                self.assertIsNone(result["corrected_fp_upper_bytes"])
                self.assertIsNone(result["mapped_storage_resident_lower_bytes"])
                self.assertEqual(result["mapped_storage_observed_resident_bytes"], 8 * P)

    def test_initial_einval_is_not_treated_as_successful_empty_walk(self):
        result = self.reader(FakeBackend()).read(123)
        self.assertFalse(result["coverage_valid"])
        self.assertIn("EINVAL before any region", result["errors"][0])

    def test_exit_or_pid_identity_reuse_invalidates_end_confirmation(self):
        result = self.reader(FakeBackend([self.item], identity_change=True)).read(123)
        self.assertFalse(result["coverage_valid"])
        self.assertIsNone(result["corrected_fp_lower_bytes"])
        self.assertIn("identity changed", result["errors"][0])

    def test_submap_is_unknown_coverage_not_zero_model_residency(self):
        hidden = dataclasses.replace(self.item, flags=1, ino=0, path="")
        result = self.reader(FakeBackend([hidden])).read(123)
        self.assertEqual(result["submap_count"], 1)
        self.assertFalse(result["coverage_valid"])
        self.assertIsNone(result["mapped_storage_resident_upper_bytes"])

    def test_source_scoped_coverage_requires_evidence_and_retains_submap_uncertainty(self):
        for evidence in (None, "", "   "):
            with self.subTest(evidence=evidence), self.assertRaisesRegex(
                    ValueError, "source evidence"):
                self.reader(FakeBackend(), mapping_scope="posix-mmap-top-level",
                            mapping_scope_evidence=evidence)
        hidden = dataclasses.replace(self.item, flags=1, ino=0, path="",
                                     address=self.item.address + 32 * P)
        result = self.reader(
            FakeBackend([self.item, hidden]), mapping_scope="posix-mmap-top-level",
            mapping_scope_evidence="Controlled loader uses POSIX mmap only").read(123)
        self.assertTrue(result["coverage_valid"])
        self.assertEqual(result["submap_count"], 1)
        self.assertEqual(result["mapped_storage_resident_lower_bytes"], 8 * P)
        self.assertEqual(result["coverage_scope"], "posix-mmap-top-level")
        self.assertIsNotNone(result["coverage_evidence"])

    def test_region_budget_and_nonadvancing_walk_are_invalid(self):
        result = self.reader(FakeBackend([self.item]), max_regions=1).read(123)
        self.assertIn("count budget", result["errors"][0])
        result = self.reader(FakeBackend([self.item, self.item])).read(123)
        self.assertIn("did not advance", result["errors"][0])
        self.assertFalse(result["coverage_valid"])

    def test_time_budget_is_invalid_not_partial_success(self):
        backend = FakeBackend([self.item])
        with patch.object(memory, "_Libproc", return_value=backend):
            reader = memory.DarwinMemoryReader([self.file], max_seconds=1e-20)
        result = reader.read(123)
        self.assertIn("time budget", result["errors"][0])
        self.assertIsNone(result["corrected_fp_upper_bytes"])

    def test_missing_ledger_returns_none_and_reason(self):
        reader = self.reader(FakeBackend(ledger_failure=True))
        self.assertIsNone(reader.read(123))
        self.assertIn("permission denied", reader.last_error)

    def test_long_runtime_walk_covers_late_model_mapping_or_fails_closed(self):
        rows = [dataclasses.replace(
            self.item, address=self.item.address + index * self.item.size,
            ino=0, path="") for index in range(400)]
        rows.append(dataclasses.replace(
            self.item, address=self.item.address + 400 * self.item.size))
        for budget, valid in ((0.15, False), (2.0, True)):
            with self.subTest(budget=budget):
                clock = [0.0]
                backend = FakeBackend(rows)
                original_region = backend.region

                def timed_region(pid, cursor):
                    clock[0] += 0.001
                    return original_region(pid, cursor)

                backend.region = timed_region
                with patch.object(memory, "_Libproc", return_value=backend), \
                     patch.object(memory.time, "monotonic", side_effect=lambda: clock[0]):
                    reader = memory.DarwinMemoryReader([self.file], max_seconds=budget)
                    result = reader.read(123)
                self.assertEqual(result["coverage_valid"], valid)
                if valid:
                    self.assertEqual(result["region_count"], 401)
                    self.assertEqual(result["corrected_fp_upper_bytes"], 10 * P)
                else:
                    self.assertIsNone(result["corrected_fp_upper_bytes"])
                    self.assertIn("time budget", result["errors"][0])

    def test_esrch_is_incomplete_until_caller_confirms_process_exit(self):
        result = self.reader(FakeBackend(
            [self.item], failure=memory.TargetUnavailable("region read size=0 errno=3")
        )).read(123)
        self.assertFalse(result["coverage_valid"])
        self.assertIsNone(result["corrected_fp_upper_bytes"])
        self.assertTrue(result["target_unavailable"])

    def test_observer_control_error_cannot_be_discarded_as_process_exit(self):
        backend = FakeBackend(
            [self.item], failure=memory.TargetUnavailable("region read size=0 errno=3"))
        original_control = backend.control

        def control(name, value=None):
            if name == "vm.self_region_footprint" and len(backend.controls) > 1:
                raise memory.CoverageError("restoration denied")
            return original_control(name, value)

        backend.control = control
        result = self.reader(backend).read(123)
        self.assertFalse(result["coverage_valid"])
        self.assertFalse(result["target_unavailable"])
        self.assertTrue(any("restoration denied" in error for error in result["errors"]))

    def test_explicit_file_identity_recognizes_hardlink_but_not_replaced_path(self):
        artifacts = memory._Artifacts([self.file])
        alias = self.root / "alias.bin"
        alias.hardlink_to(self.file)
        self.assertTrue(artifacts.match(dataclasses.replace(self.item, path=str(alias))))
        self.file.unlink()
        self.file.write_bytes(b"replacement")
        replacement_stat = self.file.stat()
        replacement = dataclasses.replace(self.item, ino=replacement_stat.st_ino)
        self.assertFalse(artifacts.match(replacement))

    def test_registered_symlinked_checkpoints_count_canonical_mappings(self):
        managed = self.root / "managed"
        store = self.root / "external-store"
        managed.mkdir()
        store.mkdir()
        weights = store / "weights.safetensors"
        weights.write_bytes(b"weights")
        (managed / "checkpoint").symlink_to(store, target_is_directory=True)
        (store / "tokenizer.bin").symlink_to(self.file)
        (store / "cycle").symlink_to(managed, target_is_directory=True)
        with patch.object(C, "MODELS_DIR", managed), \
             patch.object(C, "DATA", self.root / "data"), \
             patch.object(C, "REFERENCES", self.root / "missing.json"), \
             patch.dict(C.os.environ, {"TERMITE_GEN_MODELS_DIR": "", "TERMITE_WHISPER_CPP_MODEL": ""}):
            artifacts = memory._Artifacts(C.sampled_model_paths())
        stat = weights.stat()
        canonical = dataclasses.replace(self.item, path=str(weights),
                                        dev=stat.st_dev & 0xFFFFFFFF, ino=stat.st_ino)
        self.assertTrue(artifacts.match(canonical))
        self.assertTrue(artifacts.match(self.item))

    def test_root_matching_rejects_lookalike_prefix_and_unverified_identity(self):
        artifacts = memory._Artifacts([self.root])
        self.assertTrue(artifacts.match(self.item))
        sibling = self.root.with_name(self.root.name + "-other") / "model.bin"
        self.assertFalse(artifacts.match(dataclasses.replace(self.item, path=str(sibling))))
        with self.assertRaisesRegex(memory.CoverageError, "identity mismatch"):
            artifacts.match(dataclasses.replace(self.item, ino=self.item.ino + 1))
        with self.assertRaisesRegex(memory.CoverageError, "path unavailable"):
            artifacts.match(dataclasses.replace(self.item, path=""))


class ProcessExitObservationTests(unittest.TestCase):
    def setUp(self):
        import psutil
        self.clock = 0.0

        def sleep(seconds):
            self.clock += seconds

        monotonic = patch.object(C.time, "monotonic", side_effect=lambda: self.clock)
        sleeping = patch.object(C.time, "sleep", side_effect=sleep)
        monotonic.start()
        sleeping.start()
        self.addCleanup(monotonic.stop)
        self.addCleanup(sleeping.stop)
        self.sampler = C.Sampler.__new__(C.Sampler)
        self.process = Mock()
        self.process.create_time.return_value = 123.0
        self.process.is_running.return_value = True
        self.process.status.return_value = psutil.STATUS_RUNNING
        self.sampler.psutil = Mock(
            Process=Mock(return_value=self.process),
            NoSuchProcess=psutil.NoSuchProcess, Error=psutil.Error,
            STATUS_ZOMBIE=psutil.STATUS_ZOMBIE)
        self.value = {
            "coverage_valid": False, "target_unavailable": True,
            "errors": ["region read size=0 errno=3"],
            "corrected_fp_upper_bytes": None,
        }
        self.sampler.reader = Mock(
            read=Mock(return_value=self.value), last_target_unavailable=False,
            last_error="region read size=0 errno=3")

    def test_same_live_process_with_esrch_remains_invalid(self):
        value, missing, discarded = self.sampler._darwin_observation(123)
        self.assertIs(value, self.value)
        self.assertFalse(value["coverage_valid"])
        self.assertFalse(missing)
        self.assertIsNone(discarded)
        self.assertAlmostEqual(self.clock, self.sampler._DARWIN_EXIT_CONFIRM_SECONDS)
        self.assertEqual(value["exit_confirmation"]["create_time"], 123.0)
        self.assertEqual(value["exit_confirmation"]["final_state"], "live")
        self.assertIsNone(value["corrected_fp_upper_bytes"])

    def test_vm_disappears_before_process_becomes_zombie(self):
        self.process.status.side_effect = lambda: (
            self.sampler.psutil.STATUS_ZOMBIE if self.clock >= 0.03 else "running")
        value, missing, discarded = self.sampler._darwin_observation(123)
        self.assertIsNone(value)
        self.assertFalse(missing)
        self.assertEqual(discarded["create_time"], 123.0)
        self.assertEqual(discarded["evidence"], "original_process_is_zombie")
        self.assertEqual(discarded["exit_confirmation"]["initial_state"], "live")
        self.assertEqual(discarded["incomplete_observation"], self.value)
        self.assertFalse(discarded["incomplete_observation"]["coverage_valid"])
        self.assertIsNone(discarded["incomplete_observation"]["corrected_fp_upper_bytes"])

    def test_original_identity_disappears_during_bounded_confirmation(self):
        import psutil
        for vanished in ("reused", "absent"):
            with self.subTest(vanished=vanished):
                self.clock = 0.0

                def running():
                    if self.clock < 0.02:
                        return True
                    if vanished == "absent":
                        raise psutil.NoSuchProcess(123)
                    return False

                self.process.is_running.side_effect = running
                value, missing, discarded = self.sampler._darwin_observation(123)
                self.assertIsNone(value)
                self.assertFalse(missing)
                self.assertEqual(discarded["create_time"], 123.0)
                self.assertEqual(discarded["exit_confirmation"]["initial_state"], "live")
                self.assertEqual(discarded["evidence"], (
                    "original_process_no_longer_exists" if vanished == "absent"
                    else "original_process_exited_or_pid_reused"))

    def test_exit_after_confirmation_deadline_cannot_reclassify_live_failure(self):
        self.process.is_running.side_effect = lambda: (
            self.clock <= self.sampler._DARWIN_EXIT_CONFIRM_SECONDS)
        value, missing, discarded = self.sampler._darwin_observation(123)
        self.assertIs(value, self.value)
        self.assertFalse(missing)
        self.assertFalse(value["coverage_valid"])
        self.assertIsNone(discarded)
        self.assertEqual(value["exit_confirmation"]["final_state"], "live")
        self.clock += 0.01
        self.assertFalse(self.process.is_running())
        self.assertFalse(value["coverage_valid"])

    def test_unconfirmed_original_identity_cannot_discard_even_after_exit(self):
        import psutil
        self.process.create_time.side_effect = psutil.AccessDenied(123)
        self.process.is_running.return_value = False
        value, missing, discarded = self.sampler._darwin_observation(123)
        self.assertIs(value, self.value)
        self.assertFalse(missing)
        self.assertFalse(value["coverage_valid"])
        self.assertIsNone(discarded)

    def test_confirmed_exit_or_reuse_retains_interrupted_observation_not_a_peak(self):
        self.process.is_running.return_value = False
        value, missing, discarded = self.sampler._darwin_observation(123)
        self.assertIsNone(value)
        self.assertFalse(missing)
        self.assertEqual(discarded["create_time"], 123.0)
        self.assertEqual(discarded["evidence"], "original_process_exited_or_pid_reused")
        self.assertEqual(discarded["incomplete_observation"], self.value)

    def test_delayed_exit_retains_invalid_timeline_without_polluting_peaks(self):
        import io
        import json
        import threading

        complete = {"coverage_valid": True, "fp": 2 * P, "rss": 3 * P,
                    "mapped_storage_resident_lower_bytes": 8 * P,
                    "mapped_storage_resident_upper_bytes": 8 * P}
        self.value.update(fp=100 * P, rss=110 * P)
        self.sampler.reader.read.side_effect = [complete, self.value]
        self.process.is_running.side_effect = [True, False]
        self.sampler._pids = lambda: [123]
        self.sampler.interval = 0.2
        self.sampler.processes_seen = set()
        self.sampler.timeline = io.StringIO()
        self.sampler.stop_event = threading.Event()
        self.sampler.peak = {
            "samples": 0, "invalid_samples": 0, "coverage_errors": [],
            "discarded_exit_observations": 0, "observer_seconds": 0,
            "max_sample_gap_seconds": 0, "raw_process_tree_footprint_peak_bytes": 0,
            "peak_rss_bytes": 0, "memory_peak_lower_bytes": 0,
            "memory_peak_upper_bytes": 0,
        }

        def next_iteration(seconds):
            if self.sampler.peak["discarded_exit_observations"]:
                self.sampler.stop_event.set()

        with patch.object(C, "DARWIN", True), \
             patch.object(self.sampler.stop_event, "wait", side_effect=next_iteration):
            self.sampler._sample_loop()
        rows = [json.loads(line) for line in self.sampler.timeline.getvalue().splitlines()]
        self.assertEqual(self.sampler.peak["samples"], 1)
        self.assertEqual(self.sampler.peak["invalid_samples"], 0)
        self.assertEqual(self.sampler.peak["coverage_errors"], [])
        self.assertEqual(self.sampler.peak["raw_process_tree_footprint_peak_bytes"], 2 * P)
        self.assertEqual(self.sampler.peak["memory_peak_lower_bytes"], 8 * P)
        self.assertEqual(self.sampler.peak["memory_peak_upper_bytes"], 10 * P)
        self.assertEqual(rows[1]["observation_status"], "no_complete_process_observation")
        discarded = rows[1]["discarded_exit_observations"][0]
        self.assertEqual(discarded["create_time"], 123.0)
        self.assertEqual(discarded["exit_confirmation"]["initial_state"], "live")
        self.assertEqual(discarded["evidence"], "original_process_exited_or_pid_reused")
        self.assertFalse(discarded["incomplete_observation"]["coverage_valid"])
        self.assertIsNone(discarded["incomplete_observation"]["corrected_fp_upper_bytes"])
        self.assertEqual(discarded["incomplete_observation"]["errors"],
                         ["region read size=0 errno=3"])

    def test_zombie_is_positive_exit_evidence(self):
        self.process.status.return_value = self.sampler.psutil.STATUS_ZOMBIE
        value, missing, discarded = self.sampler._darwin_observation(123)
        self.assertIsNone(value)
        self.assertFalse(missing)
        self.assertEqual(discarded["evidence"], "original_process_is_zombie")

    def test_unreadable_liveness_cannot_discard_esrch(self):
        import psutil
        self.process.is_running.side_effect = psutil.AccessDenied(123)
        value, missing, discarded = self.sampler._darwin_observation(123)
        self.assertIs(value, self.value)
        self.assertFalse(value["coverage_valid"])
        self.assertIsNone(discarded)
        self.assertEqual(self.clock, 0.0)

    def test_dead_process_does_not_hide_a_timeout_or_unverified_submap(self):
        self.process.is_running.return_value = False
        for error in ("region walk time budget exhausted",
                      "submaps are not recursively visible through libproc"):
            self.value.update(target_unavailable=False, errors=[error])
            value, missing, discarded = self.sampler._darwin_observation(123)
            self.assertIs(value, self.value)
            self.assertFalse(value["coverage_valid"])
            self.assertIsNone(discarded)

    def test_missing_ledger_requires_typed_exit_and_positive_liveness_evidence(self):
        self.sampler.reader.read.return_value = None
        self.sampler.reader.last_target_unavailable = True
        value, missing, discarded = self.sampler._darwin_observation(123)
        self.assertIsNone(value)
        self.assertTrue(missing)
        self.assertIsNone(discarded)
        self.process.is_running.return_value = False
        value, missing, discarded = self.sampler._darwin_observation(123)
        self.assertFalse(missing)
        self.assertEqual(discarded["errors"], ["region read size=0 errno=3"])
        self.process.is_running.side_effect = lambda: self.clock < 0.23
        value, missing, discarded = self.sampler._darwin_observation(123)
        self.assertIsNone(value)
        self.assertFalse(missing)
        self.assertEqual(discarded["create_time"], 123.0)
        self.assertEqual(discarded["exit_confirmation"]["initial_state"], "live")
        self.assertEqual(discarded["errors"], ["region read size=0 errno=3"])
        self.process.is_running.side_effect = None
        self.sampler.reader.last_target_unavailable = False
        self.sampler.reader.last_error = "rusage permission denied"
        value, missing, discarded = self.sampler._darwin_observation(123)
        self.assertTrue(missing)
        self.assertIsNone(discarded)


if __name__ == "__main__":
    unittest.main()
