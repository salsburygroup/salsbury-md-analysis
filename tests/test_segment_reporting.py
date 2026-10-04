import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from salsbury_md_analysis.accepted_artifacts import reports_complete, validate_complete_report
from salsbury_md_analysis.finding_picker import (
    _compact_cross_report, _hydrogen_bond_candidates, _quality_control_records,
    _integrated_comparison_candidates, FindingPickerError, finding_sidecar_evidence,
    prioritize_findings, _hydrogen_bond_chemical_summary,
)
from salsbury_md_analysis.hydrogen_bond_reporting import occupancy_accounting, system_view
from salsbury_md_analysis.scalar_distributions import analyze_scalar_distribution, ScalarDistributionError
from salsbury_md_analysis.presentation_artifacts import validate_manifest, PresentationArtifactError


def report():
    doc = {"module_id": "hydrogen_bond_discovery", "occupancies": [],
           "candidate_dictionary": [{"bond_id": "b", "donor_atom_index": 0,
               "hydrogen_atom_index": 1, "acceptor_atom_index": 2}],
           "atom_dictionary": [], "frame_bond_matrix": []}
    # Unequal segment lengths, multiple replicas and zero-event segments.
    for system, factor in (("A", 1), ("B", 3)):
        for replica, segments in (("r1", (10, 20)), ("r2", (30, 40))):
            for index, length in enumerate(segments):
                identity = dict(system_id=system, replica_id=replica, segment_id=str(index))
                doc["frame_bond_matrix"].extend(
                    dict(identity, source_frame_index=f) for f in range(length))
                count = 0 if length == 40 else length * factor // 10
                if count:
                    doc["occupancies"].append(dict(identity, bond_id="b",
                        evaluated_frame_count=length, present_frame_count=count,
                        occupancy_fraction=count / length))
    return doc


class SegmentReportingTests(unittest.TestCase):
    def test_native_system_views_scope_inherited_occupancies(self):
        doc = report()
        doc["atom_dictionary"] = [dict(atom_index=i, identity=dict(chain_id="A",
            residue_number=i+1, residue_name="GUA", atom_name=name))
            for i, name in enumerate(("N1", "H1", "O6"))]
        doc["system_feature_spaces"] = [dict(
            system_id=system, candidate_dictionary=doc["candidate_dictionary"],
            atom_dictionary=doc["atom_dictionary"], frame_bond_matrix=[
                row for row in doc["frame_bond_matrix"] if row["system_id"] == system])
            for system in ("A", "B")]
        before = copy.deepcopy(doc)
        compact = _compact_cross_report(doc, "hydrogen_bond_discovery")
        self.assertEqual(_hydrogen_bond_candidates(doc, Path("s")),
                         _hydrogen_bond_candidates(compact, Path("s")))
        for source in (doc, compact):
            for view in source["system_feature_spaces"]:
                scoped = system_view(source, view)
                accounting = occupancy_accounting(scoped)
                self.assertEqual(accounting["totals"], {view["system_id"]: 100})
                self.assertEqual(len(accounting["segment_frame_counts"]), 4)
                _, values = _hydrogen_bond_chemical_summary(source, view["system_id"])
                self.assertAlmostEqual(next(iter(values.values())), .06 if view["system_id"] == "A" else .18)
        self.assertEqual(_hydrogen_bond_chemical_summary(doc, "A"),
                         _hydrogen_bond_chemical_summary(compact, "A"))
        self.assertEqual(doc, before)
        bad = copy.deepcopy(doc["system_feature_spaces"][0])
        bad["occupancies"] = doc["occupancies"]
        with self.assertRaisesRegex(ValueError, "another system"):
            system_view(doc, bad)
        bad.pop("occupancies")
        bad["candidate_dictionary"] = []
        with self.assertRaisesRegex(ValueError, "undeclared bond"):
            system_view(doc, bad)
        bad = copy.deepcopy(doc["system_feature_spaces"][0])
        bad["frame_bond_matrix"] = bad["frame_bond_matrix"][:-40]
        with self.assertRaisesRegex(ValueError, "omits or duplicates"):
            system_view(doc, bad)

    def test_full_compact_parity_and_zero_event_denominators(self):
        doc = report()
        compact = _compact_cross_report(doc, "hydrogen_bond_discovery")
        self.assertEqual(compact["evaluated_frame_count_by_system"], {"A": 100, "B": 100})
        self.assertEqual(len(compact["segment_frame_counts"]), 8)
        for source in (doc, compact):
            result = _hydrogen_bond_candidates(source, Path("source.json"))
            difference = next(r for r in result if r["comparison_family"].endswith("pairwise_occupancy_difference"))
            self.assertAlmostEqual(difference["effect_value"], -.12)
            self.assertTrue(all(abs(r["effect_value"]) <= 1 for r in result))
        self.assertEqual(_hydrogen_bond_candidates(doc, Path("s")), _hydrogen_bond_candidates(compact, Path("s")))

    def test_one_segment_and_all_zero_system(self):
        doc = report()
        doc["occupancies"] = [r for r in doc["occupancies"] if r["system_id"] == "A"]
        summary = occupancy_accounting(doc)
        self.assertEqual(summary["totals"]["B"], 100)
        self.assertEqual(summary["counts"].get("B", {}), {})
        one = {"occupancies": [dict(system_id="P", replica_id="r", segment_id="s",
                bond_id="x", evaluated_frame_count=4, present_frame_count=2)],
               "frame_bond_matrix": [dict(system_id="P", replica_id="r", segment_id="s") for _ in range(4)]}
        self.assertEqual(occupancy_accounting(one)["counts"], {"P": {"x": 2}})

    def test_missing_denominator_abstains_and_reports_reason(self):
        doc = report()
        del doc["frame_bond_matrix"]
        self.assertEqual(_hydrogen_bond_candidates(doc, Path("x")), [])
        self.assertEqual(_quality_control_records(doc, Path("x"))[0]["status"], "occupancy_not_estimable")

    def test_bad_counts_duplicates_and_denominator_disagreement_rejected(self):
        for change in ("duplicate", "range", "coverage"):
            doc = report()
            if change == "duplicate":
                doc["occupancies"].append(copy.deepcopy(doc["occupancies"][0]))
            elif change == "range":
                doc["occupancies"][0]["present_frame_count"] = 1000
            else:
                doc["occupancies"][0]["evaluated_frame_count"] += 1
            with self.assertRaises(ValueError):
                occupancy_accounting(doc)
        doc = _compact_cross_report(report(), "hydrogen_bond_discovery")
        doc["evaluated_frame_count_by_system"]["A"] += 1
        with self.assertRaises(ValueError):
            occupancy_accounting(doc)

    def test_legacy_integrated_occupancy_requires_reporting_rebuild(self):
        with self.assertRaisesRegex(FindingPickerError, "rebuild integration"):
            _integrated_comparison_candidates(
                {"comparison_findings": [{"module_id": "hydrogen_bond_discovery"}]}, Path("x"))
        self.assertEqual(finding_sidecar_evidence(report(), Path("x"))["finding_evidence_schema"],
                         "salsbury-finding-evidence-v4")

    def test_empty_segments_histogram_and_residence_are_independent(self):
        records = [dict(value=float(i), source_frame_index=i) for i in range(4)]
        segments = [(dict(segment_id="a"), records), (dict(segment_id="empty"), []),
                    (dict(segment_id="c"), records)]
        for static in ("0", "1"):
            with patch.dict(os.environ, {"SALSBURY_STATIC_ENSEMBLE": static}):
                dist = analyze_scalar_distribution(segments, binning_rule="scott", padding_fraction=.05)
                self.assertEqual(sum(r["count"] for r in dist["histogram"]), 8)
                self.assertEqual(dist["segment_coverage"][1]["observation_count"], 0)
                if static == "0":
                    self.assertEqual({r["segment_id"] for r in dist["residence_runs"]}, {"a", "c"})
        with patch.dict(os.environ, {"SALSBURY_STATIC_ENSEMBLE": "0"}):
            dist = analyze_scalar_distribution(segments, binning_rule="scott",
                      padding_fraction=.05, evaluate_residence=False)
            self.assertEqual(dist["residence_by_bin"], [])
            self.assertIsNone(dist["residence_runs"])
        for values in ([], [1., 1.], [1., float("nan")]):
            with self.assertRaises(ScalarDistributionError):
                analyze_scalar_distribution([({}, [dict(value=v) for v in values])],
                    binning_rule="scott", padding_fraction=.05)

    def test_legacy_sidecar_repaired_in_memory_without_changing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "results/hydrogen_bond_discovery/report.json"
            path.parent.mkdir(parents=True)
            doc = dict(report(), technical_status="complete")
            path.write_text(json.dumps(doc))
            evidence = finding_sidecar_evidence(doc, path)
            evidence["finding_evidence_schema"] = "salsbury-finding-evidence-v2"
            evidence["candidates"] = []
            evidence["cross_report_summary"]["evaluated_frame_count_by_system"] = {"A": 1, "B": 1}
            sidecar = Path(str(path) + ".summary.json")
            sidecar.write_text(json.dumps(dict(module_id=doc["module_id"], technical_status="complete",
                report_path=str(path.resolve()), report_size_bytes=path.stat().st_size,
                report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), finding_evidence=evidence)))
            before = path.read_bytes(), sidecar.read_bytes()
            result = prioritize_findings(root, write_outputs=False)
            differences = [r for r in result["all_candidates"]
                if r.get("comparison_family", "").endswith("pairwise_occupancy_difference")]
            self.assertEqual(len(differences), 1)
            self.assertAlmostEqual(differences[0]["effect_value"], -.12)
            self.assertEqual(before, (path.read_bytes(), sidecar.read_bytes()))

    def test_integrated_sidecar_cannot_hide_legacy_occupancy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "results/integrated-comparison/report.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(technical_status="complete", module_id="integrated_comparison",
                comparison_findings=[dict(module_id="hydrogen_bond_discovery")])))
            Path(str(path) + ".summary.json").write_text(json.dumps(dict(
                technical_status="complete", finding_evidence=dict(candidates=[]))))
            with self.assertRaisesRegex(FindingPickerError, "rebuild integration"):
                prioritize_findings(root, write_outputs=False)

    def test_manifest_rejects_duplicate_paths_with_distinct_ids(self):
        row = dict(artifact_id="a", artifact_type="figure", relative_path="same.svg",
                   source_report_paths=["s"], source_report_sha256=["a"*64],
                   artifact_sha256="b"*64, artifact_size_bytes=1)
        with self.assertRaisesRegex(PresentationArtifactError, "duplicate artifact path"):
            validate_manifest(dict(presentation_manifest_schema="salsbury-presentation-artifacts-v1",
                                   artifacts=[row, dict(row, artifact_id="b")]))

    def test_shared_hash_session_and_mutation_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "payload.bin"
            payload.write_bytes(b"A"*4096)
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            def write(name, data):
                p = root / name
                p.write_text(json.dumps(dict(technical_status="complete", **data)))
                return p
            leaf = write("leaf.json", dict(payload_path=str(payload), payload_sha256=digest(payload)))
            children = [write(f"c{i}.json", dict(source_report_records=[dict(path=str(leaf), sha256=digest(leaf))])) for i in range(2)]
            top = write("report.json", dict(module_id="fixture",
                source_report_records=[dict(path=str(p), sha256=digest(p)) for p in children]))
            write("report.json.summary.json", dict(module_id="fixture", report_sha256=digest(top)))
            opened = []
            original = Path.open
            def count(p, *args, **kwargs):
                if p == payload and args and args[0] == "rb":
                    opened.append(p)
                return original(p, *args, **kwargs)
            with patch.object(Path, "open", count):
                self.assertTrue(reports_complete(root, ["report.json"]))
            self.assertEqual(len(opened), 1)
            payload.write_bytes(b"B"*4096)
            self.assertFalse(reports_complete(root, ["report.json"]))
            # Validation contracts are not memoized by path.
            with self.assertRaisesRegex(ValueError, "module"):
                validate_complete_report(top, expected_module="other")
