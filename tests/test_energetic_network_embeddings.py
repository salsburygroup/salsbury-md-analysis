import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from salsbury_md_analysis.energetic_network_embeddings import (
    AMBER_CHARGE_SCALE,
    CPPTRAJ_COULOMB_FACTOR_KCAL_ANGSTROM_PER_MOL_E2,
    EnergeticNetworkError,
    compare_embedding_ensembles,
    cpptraj_style_residue_energy_matrices,
    heat_kernel,
    locally_normalized_energy_network,
    probe_energetic_parameter_source,
    read_amber_pairwise_parameters,
    read_charmm_pairwise_parameters,
    read_openmm_system_pairwise_parameters,
    energetic_network_embeddings_project,
)
from salsbury_md_analysis.atom_mapping import read_topology_atoms
from salsbury_md_analysis.quickstart import prepare_standard_analysis


def _flag(name, format_text, values, width):
    rows = [f"%FLAG {name}\n", f"%FORMAT({format_text})\n"]
    for start in range(0, len(values), 20 if width == 4 else 5):
        chunk = values[start:start + (20 if width == 4 else 5)]
        rows.append("".join(
            f"{value:<{width}}" if isinstance(value, str)
            else f"{value:{width}.8E}" if isinstance(value, float)
            else f"{value:{width}d}"
            for value in chunk
        ) + "\n")
    return "".join(rows)


def _write_dcd(path, coordinates, frame_count=20):
    def record(payload):
        marker = struct.pack("<i", len(payload))
        return marker + payload + marker

    header = bytearray(84)
    header[:4] = b"CORD"
    struct.pack_into("<3i", header, 4, frame_count, 0, 1)
    payload = record(bytes(header))
    payload += record(struct.pack("<i", 1) + b"energetic fixture".ljust(80))
    payload += record(struct.pack("<i", len(coordinates)))
    for frame in range(frame_count):
        shifted = [
            (x + (0.01 * frame if index >= 4 else 0.0), y, z)
            for index, (x, y, z) in enumerate(coordinates)
        ]
        for axis in range(3):
            payload += record(struct.pack(
                f"<{len(coordinates)}f", *(row[axis] for row in shifted)
            ))
    path.write_bytes(payload)


def _write_psf(path):
    path.write_text(
        "PSF\n\n       1 !NTITLE\n REMARKS energetic test\n"
        "       6 !NATOM\n"
        "       1 SEG 1 ALA N  N  0.200000 14.0000 0\n"
        "       2 SEG 1 ALA CA C -0.100000 12.0000 0\n"
        "       3 SEG 2 GLY N  N -0.200000 14.0000 0\n"
        "       4 SEG 2 GLY CA C  0.100000 12.0000 0\n"
        "       5 SEG 3 SER N  N  0.150000 14.0000 0\n"
        "       6 SEG 3 SER CA C -0.150000 12.0000 0\n"
        "       3 !NBOND: bonds\n"
        "       1       2       3       4       5       6\n",
        encoding="utf-8",
    )


def _write_enabled_config(path):
    path.write_text(json.dumps({
        "config_schema": "salsbury-analysis-config-v1",
        "modules": {
            "energetic_network_embeddings": {"enabled": True, "options": {}}
        },
        "views": {
            "global_common_heavy": {"enabled": False},
            "macromolecular_trace": {
                "enabled": True,
                "state_trajectory_exports_enabled": False,
            },
        },
        "execution": {
            "coordinate_cache": "off",
            "maximum_parallel_cpus": 2,
            "maximum_hours_per_cpu": 1.0,
            "maximum_memory_gib": 32.0,
            "maximum_scratch_gib": 8.0,
        },
    }), encoding="utf-8")


class EnergeticNetworkEmbeddingTests(unittest.TestCase):
    def test_charmm_and_openmm_parameter_sources_plan_and_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topology = root / "protein.pdb"
            atoms = [
                ("N", "ALA", 1, 0.0, "N"), ("CA", "ALA", 1, 1.0, "C"),
                ("N", "GLY", 2, 4.0, "N"), ("CA", "GLY", 2, 5.0, "C"),
                ("N", "SER", 3, 8.0, "N"), ("CA", "SER", 3, 9.0, "C"),
            ]
            topology.write_text("".join(
                f"ATOM  {index:5d} {name:^4s} {residue:>3s} A{number:4d}    "
                f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          {element:>2s}\n"
                for index, (name, residue, number, x, element)
                in enumerate(atoms, start=1)
            ) + "END\n", encoding="utf-8")
            trajectory = root / "trajectory.dcd"
            _write_dcd(
                trajectory, [(x, 0.0, 0.0) for _, _, _, x, _ in atoms], 50
            )
            psf = root / "protein.psf"
            _write_psf(psf)
            parameter_file = root / "protein.inp"
            parameter_file.write_text(
                "* synthetic CHARMM parameters\n*\n"
                "NONBONDED NBXMOD 5 ATOM CDIEL\n"
                "N  0.0 -0.200000 1.850000\n"
                "C  0.0 -0.100000 2.000000\n"
                "DUM 0.0  0.000000 0.000000\n"
                "NBFIX\n"
                "N C -0.150000 3.700000\n"
                "END\n",
                encoding="utf-8",
            )
            xml = root / "system.xml"
            xml.write_text(
                "<System><Forces><Force type=\"NonbondedForce\">"
                "<Particles>"
                + "".join(
                    f'<Particle q="{charge}" sig="0.35" eps="0.4184"/>'
                    for charge in (0.2, -0.1, -0.2, 0.1, 0.15, -0.15)
                )
                + "</Particles><Exceptions>"
                '<Exception p1="0" p2="5" q="0.012" sig="0.4" eps="0.8368"/>'
                "</Exceptions></Force></Forces></System>",
                encoding="utf-8",
            )
            config = root / "config.json"
            _write_enabled_config(config)

            charmm = read_charmm_pairwise_parameters(psf, [parameter_file])
            self.assertEqual(charmm.parameter_source, "charmm_psf_parameter_files_v1")
            self.assertEqual(charmm.bond_count, 3)
            self.assertEqual(charmm.nbfix_pair_type_count, 1)
            used_dummy_psf = root / "used-dummy.psf"
            used_dummy_psf.write_text(
                psf.read_text(encoding="utf-8").replace(
                    "ALA N  N  0.200000", "ALA N  DUM  0.200000", 1
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                EnergeticNetworkError,
                "Rmin/2 must be positive for used atom types: DUM",
            ):
                read_charmm_pairwise_parameters(used_dummy_psf, [parameter_file])
            n_type, c_type = charmm.atom_type_indices[:2]
            nbfix_index = charmm.nonbonded_parameter_indices[
                n_type * charmm.atom_type_count + c_type
            ]
            self.assertAlmostEqual(
                charmm.lennard_jones_a[nbfix_index], 0.15 * 3.7**12
            )

            _, topology_atoms = read_topology_atoms(topology)
            openmm = read_openmm_system_pairwise_parameters(
                xml, psf, topology_atoms
            )
            self.assertEqual(
                openmm.parameter_source, "openmm_serialized_system_xml_v1"
            )
            np.testing.assert_allclose(openmm.atom_sigma_angstrom, [3.5] * 6)
            np.testing.assert_allclose(
                openmm.atom_epsilon_kcal_per_mol, [0.1] * 6
            )
            charge_product, override_a, override_b = (
                openmm.pair_parameter_overrides[(0, 5)]
            )
            self.assertAlmostEqual(charge_product, 0.012)
            self.assertAlmostEqual(override_a, 4.0 * 0.2 * 4.0**12)
            self.assertAlmostEqual(override_b, 4.0 * 0.2 * 4.0**6)
            offset_xml = root / "system-offset.xml"
            offset_xml.write_text(
                xml.read_text(encoding="utf-8").replace(
                    "<Particles>",
                    '<ParticleOffsets><Offset parameter="lambda" particle="0" '
                    'q="1" sig="0" eps="0"/></ParticleOffsets><Particles>',
                    1,
                ),
                encoding="utf-8",
            )
            offset_probe = probe_energetic_parameter_source(
                topology, psf,
                {"format": "openmm_system_xml_v1", "files": [str(offset_xml)]},
            )
            self.assertEqual(
                offset_probe["availability_status"], "not_available"
            )
            self.assertIn("parameter offsets are unsupported", offset_probe["availability_reason"])

            cases = (
                {
                    "name": "charmm",
                    "kwargs": {"energetic_charmm_parameter_files": [parameter_file]},
                    "source": "charmm_psf_parameter_files_v1",
                },
                {
                    "name": "openmm",
                    "kwargs": {"energetic_openmm_system_xml": xml},
                    "source": "openmm_serialized_system_xml_v1",
                },
            )
            for case in cases:
                with self.subTest(case=case["name"]):
                    output = root / f"prepared-{case['name']}"
                    prepared = prepare_standard_analysis(
                        pdb_path=topology, psf_path=psf,
                        trajectories=[trajectory], output_directory=output,
                        project_id=f"{case['name']}-energetic-test",
                        frame_interval_ps=1.0, config_path=config,
                        **case["kwargs"],
                    )
                    self.assertEqual(prepared["technical_status"], "complete")
                    campaign = json.loads(
                        (output / "campaign-resource-plan.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertTrue(any(
                        row.get("module_id") == "energetic_network_embeddings"
                        for row in campaign["tasks"]
                    ))
                    report = energetic_network_embeddings_project(
                        output / "project.json"
                    )
                    self.assertEqual(report["availability_status"], "available")
                    self.assertEqual(
                        report["systems"][0]["parameter_sources"][0][
                            "parameter_source"
                        ],
                        case["source"],
                    )

    def test_gromacs_tpr_is_explicitly_unavailable_without_guessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topology = root / "one.pdb"
            topology.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000"
                "  1.00  0.00           N\nEND\n",
                encoding="utf-8",
            )
            connectivity = root / "one.json"
            connectivity.write_text(json.dumps({
                "format": "salsbury-bonds-v1", "atom_count": 1,
                "index_base": 0, "bonds": [],
            }), encoding="utf-8")
            tpr = root / "topol.tpr"
            tpr.write_bytes(b"synthetic")
            probe = probe_energetic_parameter_source(
                topology, connectivity,
                {"format": "gromacs_tpr_v1", "files": [str(tpr)]},
            )
        self.assertEqual(probe["availability_status"], "not_available")
        self.assertIn("raw GROMACS TPR extraction is not available", probe["availability_reason"])

    def test_available_amber_project_is_planned_and_runs_without_cpptraj(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topology = root / "protein.pdb"
            atoms = [
                ("N", "ALA", 1, 0.0, 0.0, 0.0, "N"),
                ("CA", "ALA", 1, 1.0, 0.0, 0.0, "C"),
                ("N", "GLY", 2, 4.0, 0.0, 0.0, "N"),
                ("CA", "GLY", 2, 5.0, 0.0, 0.0, "C"),
                ("N", "SER", 3, 8.0, 0.0, 0.0, "N"),
                ("CA", "SER", 3, 9.0, 0.0, 0.0, "C"),
            ]

            def pdb_rows(frame_shift=0.0):
                return "".join(
                    f"ATOM  {index:5d} {name:^4s} {residue:>3s} A{number:4d}    "
                    f"{x + (frame_shift if number == 3 else 0.0):8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00          {element:>2s}\n"
                    for index, (name, residue, number, x, y, z, element)
                    in enumerate(atoms, start=1)
                )

            topology.write_text(pdb_rows() + "END\n", encoding="utf-8")
            trajectory = root / "trajectory.dcd"

            def record(payload):
                marker = struct.pack("<i", len(payload))
                return marker + payload + marker

            header = bytearray(84)
            header[:4] = b"CORD"
            struct.pack_into("<3i", header, 4, 50, 0, 1)
            dcd = record(bytes(header))
            dcd += record(struct.pack("<i", 1) + b"energetic fixture".ljust(80))
            dcd += record(struct.pack("<i", len(atoms)))
            for frame in range(50):
                coordinates = [
                    (x + (frame * 0.01 if number == 3 else 0.0), y, z)
                    for _, _, number, x, y, z, _ in atoms
                ]
                for axis in range(3):
                    dcd += record(struct.pack(
                        f"<{len(atoms)}f", *(row[axis] for row in coordinates)
                    ))
            trajectory.write_bytes(dcd)
            prmtop = root / "protein.prmtop"
            prmtop_text = "%VERSION VERSION_STAMP = V0001.000\n"
            prmtop_text += _flag("POINTERS", "10I8", [6, 1], 8)
            prmtop_text += _flag(
                "ATOM_NAME", "20a4", ["N", "CA", "N", "CA", "N", "CA"], 4
            )
            prmtop_text += _flag(
                "CHARGE", "5E16.8",
                [value * AMBER_CHARGE_SCALE for value in (0.2, -0.1, -0.2, 0.1, 0.15, -0.15)],
                16,
            )
            prmtop_text += _flag("ATOM_TYPE_INDEX", "10I8", [1] * 6, 8)
            prmtop_text += _flag("NONBONDED_PARM_INDEX", "10I8", [1], 8)
            prmtop_text += _flag("RESIDUE_LABEL", "20a4", ["ALA", "GLY", "SER"], 4)
            prmtop_text += _flag("RESIDUE_POINTER", "10I8", [1, 3, 5], 8)
            prmtop_text += _flag("LENNARD_JONES_ACOEF", "5E16.8", [0.0], 16)
            prmtop_text += _flag("LENNARD_JONES_BCOEF", "5E16.8", [0.0], 16)
            prmtop_text += _flag("NUMBER_EXCLUDED_ATOMS", "10I8", [1] * 6, 8)
            prmtop_text += _flag("EXCLUDED_ATOMS_LIST", "10I8", [2, 0, 4, 0, 6, 0], 8)
            prmtop_text += _flag("BONDS_INC_HYDROGEN", "10I8", [], 8)
            prmtop_text += _flag(
                "BONDS_WITHOUT_HYDROGEN", "10I8",
                [0, 3, 1, 6, 9, 1, 12, 15, 1], 8,
            )
            prmtop.write_text(prmtop_text, encoding="utf-8")
            config = root / "config.json"
            config.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "modules": {
                    "energetic_network_embeddings": {"enabled": True, "options": {}}
                },
                "views": {
                    "global_common_heavy": {"enabled": False},
                    "macromolecular_trace": {
                        "enabled": True,
                        "state_trajectory_exports_enabled": False,
                    },
                },
                "execution": {
                    "coordinate_cache": "off",
                    "maximum_parallel_cpus": 2,
                    "maximum_hours_per_cpu": 1.0,
                    "maximum_memory_gib": 32.0,
                    "maximum_scratch_gib": 8.0,
                },
            }), encoding="utf-8")
            output = root / "prepared"
            prepared = prepare_standard_analysis(
                pdb_path=topology, psf_path=prmtop,
                trajectories=[trajectory], output_directory=output,
                project_id="amber-energetic-test", frame_interval_ps=1.0,
                config_path=config,
            )
            self.assertEqual(prepared["technical_status"], "complete")
            campaign = json.loads(
                (output / "campaign-resource-plan.json").read_text(encoding="utf-8")
            )
            task = next(
                row for row in campaign["tasks"]
                if row.get("module_id") == "energetic_network_embeddings"
            )
            report = energetic_network_embeddings_project(output / "project.json")
        self.assertEqual(task["task_scope"], "direct_trajectory_estimator")
        self.assertEqual(report["availability_status"], "available")
        self.assertTrue(report["analysis_performed"])
        self.assertEqual(len(report["nodes"]), 3)
        self.assertEqual(report["systems"][0]["evaluated_frame_count"], 50)
        self.assertEqual(report["systems"][0]["vdw_negligibility_status"], "passed")
        self.assertFalse(report["pair_energy_contract"]["complete_pme_decomposition_claimed"])

    def test_amber_parameter_reader_preserves_parameters_and_cpptraj_exclusions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "five-atom.prmtop"
            text = "%VERSION VERSION_STAMP = V0001.000\n"
            text += _flag("POINTERS", "10I8", [5, 1], 8)
            text += _flag(
                "ATOM_NAME", "20a4", ["A1", "A2", "A3", "A4", "A5"], 4
            )
            text += _flag(
                "CHARGE", "5E16.8",
                [AMBER_CHARGE_SCALE, -AMBER_CHARGE_SCALE, 0.0, 0.0, 0.0], 16,
            )
            text += _flag("ATOM_TYPE_INDEX", "10I8", [1] * 5, 8)
            text += _flag("NONBONDED_PARM_INDEX", "10I8", [1], 8)
            text += _flag(
                "RESIDUE_LABEL", "20a4", ["ALA", "GLY", "SER", "VAL", "LEU"], 4
            )
            text += _flag("RESIDUE_POINTER", "10I8", [1, 2, 3, 4, 5], 8)
            text += _flag("LENNARD_JONES_ACOEF", "5E16.8", [64.0], 16)
            text += _flag("LENNARD_JONES_BCOEF", "5E16.8", [16.0], 16)
            text += _flag("BONDS_INC_HYDROGEN", "10I8", [], 8)
            text += _flag(
                "BONDS_WITHOUT_HYDROGEN", "10I8",
                [0, 3, 1, 3, 6, 1, 6, 9, 1, 9, 12, 1], 8,
            )
            path.write_text(text, encoding="utf-8")
            parameters = read_amber_pairwise_parameters(path)
        np.testing.assert_allclose(parameters.charges_e, [1.0, -1.0, 0.0, 0.0, 0.0])
        self.assertEqual(parameters.residue_names, ("ALA", "GLY", "SER", "VAL", "LEU"))
        self.assertIn((0, 1), parameters.excluded_pairs)  # 1-2
        self.assertIn((0, 2), parameters.excluded_pairs)  # 1-3
        self.assertIn((0, 3), parameters.excluded_pairs)  # 1-4
        self.assertNotIn((0, 4), parameters.excluded_pairs)
        self.assertEqual(parameters.bond_count, 4)
        np.testing.assert_allclose(parameters.lennard_jones_a, [64.0])

    def test_cpptraj_pair_equations_and_absolute_residue_aggregation(self):
        terms = {
            "left": np.asarray([0]), "right": np.asarray([1]),
            "residue_left": np.asarray([0]), "residue_right": np.asarray([1]),
            "charge_product": np.asarray([-1.0]),
            "lj_a": np.asarray([64.0]), "lj_b": np.asarray([16.0]),
        }
        electrostatic, vdw = cpptraj_style_residue_energy_matrices(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], terms, 2,
        )
        self.assertAlmostEqual(
            electrostatic[0, 1],
            CPPTRAJ_COULOMB_FACTOR_KCAL_ANGSTROM_PER_MOL_E2 / 2.0,
        )
        self.assertAlmostEqual(vdw[0, 1], abs(64.0 / 2.0**12 - 16.0 / 2.0**6))
        self.assertAlmostEqual(electrostatic[1, 0], electrostatic[0, 1])

    def test_local_normalization_threshold_and_heat_kernel(self):
        energy = np.asarray([
            [0.0, 4.0, 2.0],
            [4.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ])
        network = locally_normalized_energy_network(energy, threshold=0.5)
        self.assertGreater(network[0, 1], 0.0)
        self.assertGreater(network[0, 2], 0.0)
        self.assertEqual(network[1, 2], 0.0)
        kernel = heat_kernel(network, diffusion_time=6.0)
        self.assertEqual(kernel.shape, (3, 3))
        np.testing.assert_allclose(kernel, kernel.T)

    def test_wasserstein_comparison_is_zero_for_identical_ensembles(self):
        kernel = heat_kernel(
            [[0.0, 1.0, 0.5], [1.0, 0.0, 0.2], [0.5, 0.2, 0.0]], 6.0
        )
        comparisons = compare_embedding_ensembles({
            "a": [kernel, kernel], "b": [kernel, kernel],
        })
        self.assertEqual(len(comparisons), 1)
        self.assertTrue(all(
            row["summed_wasserstein_distance"] == 0.0
            for row in comparisons[0]["residue_distances"]
        ))


if __name__ == "__main__":
    unittest.main()
