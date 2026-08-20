import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.periodic import load_connectivity


ROOT = Path(__file__).resolve().parents[1]


def _load_system_exporter():
    path = ROOT / "scripts" / "export_openmm_system_connectivity.py"
    spec = importlib.util.spec_from_file_location("openmm_system_connectivity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load OpenMM system-connectivity exporter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atom(serial, name, residue_number, x, element):
    return (
        f"HETATM{serial:5d} {name:^4s} HOH A{residue_number:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          {element:>2s}\n"
    )


class OpenMMConnectivityExportTests(unittest.TestCase):
    def test_custom_bond_forces_are_explicit_opt_in(self):
        exporter = _load_system_exporter()

        class HarmonicBondForce:
            def getNumBonds(self):
                return 1

            def getBondParameters(self, index):
                self.assert_index(index)
                return 0, 1, 0.1, 100.0

            @staticmethod
            def assert_index(index):
                if index != 0:
                    raise IndexError(index)

        class CustomBondForce:
            def getNumBonds(self):
                return 1

            def getBondParameters(self, index):
                if index != 0:
                    raise IndexError(index)
                return 2, 8, []

        class FakeSystem:
            def getForces(self):
                return [HarmonicBondForce(), CustomBondForce()]

        system = FakeSystem()
        self.assertEqual(list(exporter._force_bonds(system)), [(0, 1)])
        self.assertEqual(
            list(exporter._force_bonds(system, include_custom_bonds=True)),
            [(0, 1), (2, 8)],
        )

    def test_constraint_only_pairs_are_explicit_opt_in(self):
        exporter = _load_system_exporter()
        topology_bonds = {(0, 1), (1, 2)}
        harmonic_bonds = {(0, 1), (3, 4)}
        custom_bonds = {(1, 2), (7, 8)}
        constraints = {(0, 1), (1, 2), (2, 4), (5, 6)}

        selected, categories = exporter._select_bonds(
            topology_bonds, harmonic_bonds, custom_bonds, constraints
        )
        self.assertEqual(selected, topology_bonds)
        self.assertEqual(categories["harmonic_force_only"], {(3, 4)})
        self.assertEqual(categories["custom_force_only"], {(7, 8)})
        self.assertEqual(categories["constraint_only"], {(2, 4), (5, 6)})

        selected_with_system_pairs, repeated_categories = (
            exporter._select_bonds(
                topology_bonds,
                harmonic_bonds,
                custom_bonds,
                constraints,
                include_harmonic_force_only_pairs=True,
                include_custom_bonds=True,
                include_constraint_only_pairs=True,
            )
        )
        self.assertEqual(
            selected_with_system_pairs,
            {(0, 1), (1, 2), (2, 4), (3, 4), (5, 6), (7, 8)},
        )
        self.assertEqual(repeated_categories, categories)

    @unittest.skipUnless(
        importlib.util.find_spec("openmm") is not None,
        "optional OpenMM connectivity dependency is unavailable",
    )
    def test_export_is_loadable_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdb = root / "water.pdb"
            pdb.write_text(
                "".join((
                    _atom(1, "O", 1, 0.0, "O"),
                    _atom(2, "H1", 1, 0.96, "H"),
                    _atom(3, "H2", 1, -0.24, "H"),
                    "END\n",
                )),
                encoding="utf-8",
            )
            output = root / "water.bonds.json"
            process = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "export_openmm_connectivity.py"),
                    str(pdb),
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["atom_count"], 3)
            self.assertEqual(len(payload["bonds"]), 2)
            self.assertIn("openmm_version", payload["provenance"])
            bonds, identity = load_connectivity(output, 3)
            self.assertEqual(len(bonds), 2)
            self.assertEqual(identity["bond_count"], 2)


if __name__ == "__main__":
    unittest.main()
