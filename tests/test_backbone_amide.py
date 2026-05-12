from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from HBOND_CHEMEM.cli import build_parser, main
from HBOND_CHEMEM.backbone_amide import (
    PEPTIDE_N_TYPE,
    PEPTIDE_O_TYPE,
    chemem_hbond_score,
    score_pdb,
    score_structure,
    write_csv,
    write_json,
)
from HBOND_CHEMEM.environment_context import CONTEXT_FIELDS
from HBOND_CHEMEM.hydrogen import (
    _charmm_template_hydrogen_variants,
    _lookup_ccd_component,
    add_hydrogens_with_pdbfixer,
)
from HBOND_CHEMEM.pdb_io import find_backbone_donors, parse_pdb
from HBOND_CHEMEM.protein_hbond_typing import parse_atom_selector
from HBOND_CHEMEM.reference_hbond_score import eval_poly, load_tables
from HBOND_CHEMEM.score_bounds import load_score_bounds, normalize_hbond_score


def pdb_atom(
    serial: int,
    name: str,
    res_name: str,
    chain: str,
    res_seq: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    return (
        f"ATOM  {serial:5d} {name:<4s} {res_name:>3s} {chain:1s}{res_seq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{20.00:6.2f}          {element:>2s}"
    )


def write_temp_pdb(lines: list[str]) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False)
    with handle:
        handle.write("\n".join(lines) + "\n")
    return Path(handle.name)


def simple_hbond_lines(extra_atoms: list[str] | None = None) -> list[str]:
    lines = [
        pdb_atom(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
        pdb_atom(2, "H", "ALA", "A", 1, 1.0, 0.0, 0.0, "H"),
        pdb_atom(3, "CA", "ALA", "A", 1, 0.0, 1.0, 0.0, "C"),
        pdb_atom(4, "O", "GLY", "A", 2, 2.8, 0.0, 0.0, "O"),
    ]
    if extra_atoms:
        lines.extend(extra_atoms)
    return lines


def h_shell_atoms(start_serial: int = 10) -> list[str]:
    coords = [
        (1.0, 3.0, 0.0),
        (1.0, -3.0, 0.0),
        (1.0, 0.0, 3.0),
        (1.0, 0.0, -3.0),
        (3.1, 2.1, 0.0),
        (3.1, -2.1, 0.0),
        (-1.1, 2.1, 0.0),
        (-1.1, -2.1, 0.0),
        (1.0, 2.1, 2.1),
        (1.0, -2.1, 2.1),
        (1.0, 2.1, -2.1),
        (1.0, -2.1, -2.1),
    ]
    return [
        pdb_atom(start_serial + index, "CB", "ALA", "A", 10 + index, x, y, z, "C")
        for index, (x, y, z) in enumerate(coords)
    ]


def two_hbond_lines() -> list[str]:
    return [
        pdb_atom(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
        pdb_atom(2, "H", "ALA", "A", 1, 1.0, 0.0, 0.0, "H"),
        pdb_atom(3, "CA", "ALA", "A", 1, 0.0, 1.0, 0.0, "C"),
        pdb_atom(4, "O", "GLY", "A", 2, 2.8, 0.0, 0.0, "O"),
        pdb_atom(20, "OG", "SER", "A", 20, 20.0, 0.0, 0.0, "O"),
        pdb_atom(21, "HG", "SER", "A", 20, 21.0, 0.0, 0.0, "H"),
        pdb_atom(22, "OD1", "ASP", "A", 21, 22.8, 0.0, 0.0, "O"),
    ]


def competing_acceptor_lines() -> list[str]:
    return [
        pdb_atom(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
        pdb_atom(2, "H", "ALA", "A", 1, 1.0, 0.0, 0.0, "H"),
        pdb_atom(3, "CA", "ALA", "A", 1, 0.0, 1.0, 0.0, "C"),
        pdb_atom(4, "O", "GLY", "A", 2, 2.8, 0.0, 0.0, "O"),
        pdb_atom(5, "O", "SER", "A", 3, 2.4, 1.0, 0.0, "O"),
    ]


def two_hydrogen_donor_lines() -> list[str]:
    return [
        pdb_atom(1, "NZ", "LYS", "A", 1, 0.0, 0.0, 0.0, "N"),
        pdb_atom(2, "HZ1", "LYS", "A", 1, 1.0, 0.0, 0.0, "H"),
        pdb_atom(3, "HZ2", "LYS", "A", 1, -1.0, 0.0, 0.0, "H"),
        pdb_atom(4, "O", "GLY", "A", 2, 2.8, 0.0, 0.0, "O"),
        pdb_atom(5, "O", "SER", "A", 3, -2.8, 0.0, 0.0, "O"),
    ]


def module(name: str, **attrs) -> ModuleType:
    value = ModuleType(name)
    for key, attr_value in attrs.items():
        setattr(value, key, attr_value)
    return value


class FakeOpenMMElement:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol


class FakeOpenMMAtom:
    def __init__(self, name: str, symbol: str) -> None:
        self.name = name
        self.element = FakeOpenMMElement(symbol)


class FakeOpenMMChain:
    def __init__(self, chain_id: str = "A") -> None:
        self.id = chain_id


class FakeOpenMMResidue:
    def __init__(self, name: str, atoms: list[FakeOpenMMAtom], index: int = 0) -> None:
        self.name = name
        self.index = index
        self.id = str(index + 1)
        self.insertionCode = ""
        self.chain = FakeOpenMMChain()
        self._atoms = atoms

    def atoms(self):
        return iter(self._atoms)


class FakeOpenMMTopology:
    def __init__(self, residues: list[FakeOpenMMResidue]) -> None:
        self._residues = residues

    def residues(self):
        return iter(self._residues)


class FakeOpenMMTemplate:
    def __init__(self, atoms: list[FakeOpenMMAtom], bonds: list[tuple[int, int]]) -> None:
        self.atoms = atoms
        self.bonds = bonds
        self.atomIndices = {atom.name: index for index, atom in enumerate(atoms)}


class BackboneAmideTests(unittest.TestCase):
    def test_uncapped_peptide_score_can_exceed_old_rep_cap(self) -> None:
        tables = load_tables()
        angle = 180.0
        distance = 2.0
        a_value = eval_poly(tables.poly_a[str(PEPTIDE_N_TYPE)][str(PEPTIDE_O_TYPE)], angle)
        b_value = eval_poly(tables.poly_b[str(PEPTIDE_N_TYPE)][str(PEPTIDE_O_TYPE)], angle)
        c_value = eval_poly(tables.poly_c[str(PEPTIDE_N_TYPE)][str(PEPTIDE_O_TYPE)], angle)

        score = chemem_hbond_score(a_value, b_value, c_value, distance)

        self.assertGreater(score, 5.0)

    def test_peptide_normalization_maps_favorable_strength_to_zero_one(self) -> None:
        tables = load_tables()
        bounds = load_score_bounds()
        pair_bound = bounds["pairs"][str(PEPTIDE_N_TYPE)][str(PEPTIDE_O_TYPE)]
        angle = pair_bound["angle_at_strongest_favorable"]
        distance = pair_bound["distance_at_strongest_favorable"]
        a_value = eval_poly(tables.poly_a[str(PEPTIDE_N_TYPE)][str(PEPTIDE_O_TYPE)], angle)
        b_value = eval_poly(tables.poly_b[str(PEPTIDE_N_TYPE)][str(PEPTIDE_O_TYPE)], angle)
        c_value = eval_poly(tables.poly_c[str(PEPTIDE_N_TYPE)][str(PEPTIDE_O_TYPE)], angle)

        strongest_score = chemem_hbond_score(a_value, b_value, c_value, distance)
        normalized = normalize_hbond_score(strongest_score, PEPTIDE_N_TYPE, PEPTIDE_O_TYPE, bounds)
        repulsive = normalize_hbond_score(10.0, PEPTIDE_N_TYPE, PEPTIDE_O_TYPE, bounds)

        self.assertAlmostEqual(normalized, 1.0, places=12)
        self.assertEqual(repulsive, 0.0)
        self.assertGreaterEqual(normalized, 0.0)
        self.assertLessEqual(normalized, 1.0)

    def test_score_bounds_match_polynomial_table_keys(self) -> None:
        tables = load_tables()
        bounds = load_score_bounds()

        self.assertEqual(set(bounds["pairs"]), set(tables.poly_a))
        for donor_key, acceptor_bounds in bounds["pairs"].items():
            self.assertEqual(set(acceptor_bounds), set(tables.poly_a[donor_key]))
            for acceptor_key in acceptor_bounds:
                self.assertIn(acceptor_key, tables.poly_b[donor_key])
                self.assertIn(acceptor_key, tables.poly_c[donor_key])

    def test_parser_preserves_ids_and_finds_explicit_backbone_hydrogen(self) -> None:
        path = write_temp_pdb(
            [
                pdb_atom(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
                pdb_atom(2, "H", "ALA", "A", 1, 1.0, 0.0, 0.0, "H"),
                pdb_atom(3, "CA", "ALA", "A", 1, 0.0, 1.0, 0.0, "C"),
                pdb_atom(4, "O", "GLY", "A", 2, 2.8, 0.0, 0.0, "O"),
            ]
        )
        structure = parse_pdb(path)
        donors = find_backbone_donors(structure)

        self.assertEqual(donors[0].atom.serial, 1)
        self.assertEqual(donors[0].hydrogens[0].serial, 2)
        self.assertEqual(donors[0].atom.chain_id, "A")
        self.assertEqual(donors[0].atom.res_seq, 1)

    def test_generated_hydrogen_mapping_uses_synthetic_id(self) -> None:
        original = parse_pdb(
            write_temp_pdb(
                [
                    pdb_atom(10, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
                    pdb_atom(11, "O", "GLY", "A", 2, 2.8, 0.0, 0.0, "O"),
                ]
            )
        )
        hydrated = parse_pdb(
            write_temp_pdb(
                [
                    pdb_atom(100, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
                    pdb_atom(101, "H", "ALA", "A", 1, 1.0, 0.0, 0.0, "H"),
                    pdb_atom(102, "O", "GLY", "A", 2, 2.8, 0.0, 0.0, "O"),
                ]
            )
        )

        merged = hydrated.with_original_heavy_atom_ids(original)
        donors = find_backbone_donors(merged)

        self.assertEqual(donors[0].atom.atom_id, 10)
        self.assertTrue(donors[0].hydrogens[0].generated)
        self.assertEqual(donors[0].hydrogens[0].atom_id, "generated:model:1:chain:A:1:_:H")

    def test_candidate_detection_applies_only_distance_and_angle_cutoffs(self) -> None:
        path = write_temp_pdb(
            [
                pdb_atom(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
                pdb_atom(2, "H", "ALA", "A", 1, 1.0, 0.0, 0.0, "H"),
                pdb_atom(3, "O", "GLY", "A", 2, 2.8, 0.0, 0.0, "O"),
                pdb_atom(4, "O", "SER", "A", 3, 6.1, 0.0, 0.0, "O"),
                pdb_atom(5, "O", "THR", "A", 4, 0.5, 1.0, 0.0, "O"),
                pdb_atom(6, "N", "PRO", "A", 5, 10.0, 0.0, 0.0, "N"),
                pdb_atom(7, "O", "PRO", "A", 5, 10.5, 0.0, 0.0, "O"),
            ]
        )

        hbonds = score_structure(parse_pdb(path))

        self.assertEqual(len(hbonds), 1)
        self.assertEqual(hbonds[0].donor.name, "N")
        self.assertEqual(hbonds[0].acceptor.serial, 3)
        self.assertGreater(hbonds[0].dha_angle, 110.0)

    def test_hbond_distance_cutoff_defaults_to_3p5_and_is_configurable(self) -> None:
        path = write_temp_pdb(
            [
                pdb_atom(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
                pdb_atom(2, "H", "ALA", "A", 1, 1.0, 0.0, 0.0, "H"),
                pdb_atom(3, "O", "GLY", "A", 2, 4.0, 0.0, 0.0, "O"),
            ]
        )
        structure = parse_pdb(path)

        default_hbonds = score_structure(structure, context_mode="none")
        relaxed_hbonds = score_structure(structure, context_mode="none", distance_cutoff=4.5)

        self.assertEqual(default_hbonds, [])
        self.assertEqual(len(relaxed_hbonds), 1)
        self.assertAlmostEqual(relaxed_hbonds[0].donor_acceptor_distance, 4.0)

    def test_one_hbond_per_donor_hydrogen_can_select_best_distance(self) -> None:
        structure = parse_pdb(write_temp_pdb(competing_acceptor_lines()))

        all_hbonds = score_structure(structure, context_mode="none")
        selected = score_structure(
            structure,
            context_mode="none",
            hbond_per_donor_hydrogen="best-distance",
        )

        self.assertEqual(len(all_hbonds), 2)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].acceptor.res_seq, 3)
        self.assertLess(selected[0].donor_acceptor_distance, 2.7)

    def test_one_hbond_per_donor_hydrogen_can_select_best_normalized_score(self) -> None:
        structure = parse_pdb(write_temp_pdb(competing_acceptor_lines()))

        selected = score_structure(
            structure,
            context_mode="none",
            hbond_per_donor_hydrogen="best-normalized-score",
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].acceptor.res_seq, 2)
        self.assertGreater(selected[0].normalized_score, 0.8)

    def test_one_hbond_per_donor_hydrogen_groups_by_hydrogen_not_heavy_donor(self) -> None:
        structure = parse_pdb(write_temp_pdb(two_hydrogen_donor_lines()))

        selected = score_structure(
            structure,
            atom_types="ALL",
            context_mode="none",
            hbond_per_donor_hydrogen="best-distance",
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual({hbond.hydrogen.name for hbond in selected}, {"HZ1", "HZ2"})
        self.assertEqual({hbond.acceptor.res_seq for hbond in selected}, {2, 3})

    def test_atom_selectors_filter_by_participating_atom(self) -> None:
        path = write_temp_pdb(
            [
                pdb_atom(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
                pdb_atom(2, "H", "ALA", "A", 1, 1.0, 0.0, 0.0, "H"),
                pdb_atom(3, "O", "GLY", "A", 2, 2.8, 0.0, 0.0, "O"),
                pdb_atom(4, "OG", "SER", "A", 3, 20.0, 0.0, 0.0, "O"),
                pdb_atom(5, "HG", "SER", "A", 3, 21.0, 0.0, 0.0, "H"),
                pdb_atom(6, "OD1", "ASP", "A", 4, 22.8, 0.0, 0.0, "O"),
            ]
        )
        structure = parse_pdb(path)

        default_hbonds = score_structure(structure)
        n_hbonds = score_structure(structure, atom_types="N")
        o_hbonds = score_structure(structure, atom_types="O")
        n_o_hbonds = score_structure(structure, atom_types="N,O")
        all_hbonds = score_structure(structure, atom_types="ALL")
        wildcard_hbonds = score_structure(structure, atom_types="*")
        ser_og_hbonds = score_structure(structure, atom_types="SER:OG")
        type_19_hbonds = score_structure(structure, atom_types="19")
        type_40_hbonds = score_structure(structure, atom_types="40")

        self.assertEqual(len(default_hbonds), 1)
        self.assertEqual(len(n_hbonds), 1)
        self.assertEqual(n_hbonds[0].donor.name, "N")
        self.assertNotIn(("OG", "OD1"), {(h.donor.name, h.acceptor.name) for h in n_hbonds})
        self.assertNotIn(("OG", "OD1"), {(h.donor.name, h.acceptor.name) for h in o_hbonds})
        self.assertEqual([(h.donor.name, h.acceptor.name) for h in o_hbonds], [("N", "O")])
        self.assertEqual(len(n_o_hbonds), 1)
        self.assertEqual(len(all_hbonds), 2)
        self.assertEqual(len(wildcard_hbonds), 2)
        self.assertEqual([(h.donor.name, h.acceptor.name) for h in ser_og_hbonds], [("OG", "OD1")])
        self.assertEqual([(h.donor.name, h.acceptor.name) for h in type_19_hbonds], [("OG", "OD1")])
        self.assertEqual([(h.donor.name, h.acceptor.name) for h in type_40_hbonds], [("N", "O")])
        for hbond in all_hbonds:
            self.assertGreaterEqual(hbond.normalized_score, 0.0)
            self.assertLessEqual(hbond.normalized_score, 1.0)

    def test_bare_atom_selector_matches_atom_name_not_element(self) -> None:
        structure = parse_pdb(
            write_temp_pdb(
                [
                    pdb_atom(1, "NZ", "LYS", "A", 1, 0.0, 0.0, 0.0, "N"),
                    pdb_atom(2, "O", "GLY", "A", 2, 2.8, 0.0, 0.0, "O"),
                ]
            )
        )
        nz_atom = structure.atoms[0]
        o_atom = structure.atoms[1]

        self.assertFalse(parse_atom_selector("N").matches(nz_atom, 13))
        self.assertFalse(parse_atom_selector("O").matches(nz_atom, 13))
        self.assertTrue(parse_atom_selector("NZ").matches(nz_atom, 13))
        self.assertTrue(parse_atom_selector("O").matches(o_atom, 39))

    def test_cli_style_outputs_have_matching_json_and_csv_rows(self) -> None:
        input_path = write_temp_pdb(
            [
                pdb_atom(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
                pdb_atom(2, "H", "ALA", "A", 1, 1.0, 0.0, 0.0, "H"),
                pdb_atom(3, "O", "GLY", "A", 2, 2.8, 0.0, 0.0, "O"),
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "hbonds.json"
            csv_path = Path(tmpdir) / "hbonds.csv"
            result = score_pdb(input_path, hydrogen_mode="explicit")
            write_json(result, json_path)
            write_csv(result, csv_path)

            data = json.loads(json_path.read_text())
            with csv_path.open() as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(data["hbonds"]), 1)
        self.assertEqual(len(rows), 1)
        self.assertNotIn("weighted_score", data["hbonds"][0])
        self.assertNotIn("weighted_score", rows[0])
        self.assertIn("normalized_score", data["hbonds"][0])
        self.assertIn("normalized_score", rows[0])
        self.assertGreaterEqual(data["hbonds"][0]["normalized_score"], 0.0)
        self.assertLessEqual(data["hbonds"][0]["normalized_score"], 1.0)
        self.assertEqual(int(rows[0]["donor_atom_id"]), data["hbonds"][0]["donor_atom_id"])
        self.assertEqual(int(rows[0]["acceptor_atom_id"]), data["hbonds"][0]["acceptor_atom_id"])

    def test_fast_context_marks_buried_hydrogen_as_less_accessible(self) -> None:
        exposed = score_structure(parse_pdb(write_temp_pdb(simple_hbond_lines())))[0].to_row()
        buried = score_structure(
            parse_pdb(write_temp_pdb(simple_hbond_lines(h_shell_atoms())))
        )[0].to_row()

        self.assertLess(buried["env_h_sasa_fraction"], exposed["env_h_sasa_fraction"])
        self.assertLess(
            buried["env_h_solvent_reach_fraction"],
            exposed["env_h_solvent_reach_fraction"],
        )
        self.assertGreater(buried["env_h_packing_count_6p5"], exposed["env_h_packing_count_6p5"])

    def test_fast_context_tracks_electrostatic_sign_and_hydrophobic_neighbors(self) -> None:
        positive_extra = [pdb_atom(5, "NZ", "LYS", "A", 3, 1.0, 3.0, 0.0, "N")]
        negative_extra = [pdb_atom(5, "OD1", "ASP", "A", 3, 1.0, 3.0, 0.0, "O")]
        hydrophobic_extra = [
            pdb_atom(5, "CD1", "LEU", "A", 3, 1.0, 3.0, 0.0, "C"),
            pdb_atom(6, "CD2", "LEU", "A", 3, 1.0, -3.0, 0.0, "C"),
            pdb_atom(7, "CG", "LEU", "A", 3, 1.0, 0.0, 3.0, "C"),
        ]

        positive = score_structure(parse_pdb(write_temp_pdb(simple_hbond_lines(positive_extra))))[
            0
        ].to_row()
        negative = score_structure(parse_pdb(write_temp_pdb(simple_hbond_lines(negative_extra))))[
            0
        ].to_row()
        bare = score_structure(parse_pdb(write_temp_pdb(simple_hbond_lines())))[0].to_row()
        hydrophobic = score_structure(
            parse_pdb(write_temp_pdb(simple_hbond_lines(hydrophobic_extra)))
        )[0].to_row()

        self.assertGreater(positive["env_h_electrostatic"], 0.0)
        self.assertLess(negative["env_h_electrostatic"], 0.0)
        self.assertGreater(positive["env_h_electrostatic"], negative["env_h_electrostatic"])
        self.assertGreater(hydrophobic["env_h_hydrophobic"], bare["env_h_hydrophobic"])

    def test_parallel_scoring_matches_serial_rows(self) -> None:
        structure = parse_pdb(write_temp_pdb(two_hbond_lines()))

        serial = score_structure(
            structure,
            atom_types="ALL",
            context_mode="none",
            workers=1,
        )
        parallel = score_structure(
            structure,
            atom_types="ALL",
            context_mode="none",
            workers=2,
        )
        capped = score_structure(
            structure,
            atom_types="ALL",
            context_mode="none",
            workers=100,
        )

        serial_rows = [hbond.to_row() for hbond in serial]
        self.assertEqual(serial_rows, [hbond.to_row() for hbond in parallel])
        self.assertEqual(serial_rows, [hbond.to_row() for hbond in capped])

    def test_parallel_context_matches_serial_rows(self) -> None:
        structure = parse_pdb(write_temp_pdb(two_hbond_lines()))

        serial = score_structure(structure, atom_types="ALL", workers=1)
        parallel = score_structure(structure, atom_types="ALL", workers=2)

        self.assertEqual([hbond.to_row() for hbond in serial], [hbond.to_row() for hbond in parallel])
        for hbond in parallel:
            self.assertIsNotNone(hbond.environment_context)

    def test_context_schema_is_written_for_fast_and_none_modes(self) -> None:
        input_path = write_temp_pdb(simple_hbond_lines())
        with tempfile.TemporaryDirectory() as tmpdir:
            fast_json_path = Path(tmpdir) / "fast.json"
            fast_csv_path = Path(tmpdir) / "fast.csv"
            none_json_path = Path(tmpdir) / "none.json"
            none_csv_path = Path(tmpdir) / "none.csv"

            fast_result = score_pdb(input_path, hydrogen_mode="explicit")
            none_result = score_pdb(input_path, hydrogen_mode="explicit", context_mode="none")
            write_json(fast_result, fast_json_path)
            write_csv(fast_result, fast_csv_path)
            write_json(none_result, none_json_path)
            write_csv(none_result, none_csv_path)

            fast_data = json.loads(fast_json_path.read_text())
            none_data = json.loads(none_json_path.read_text())
            with fast_csv_path.open() as handle:
                fast_rows = list(csv.DictReader(handle))
            with none_csv_path.open() as handle:
                none_rows = list(csv.DictReader(handle))

        self.assertEqual(fast_data["metadata"]["context_mode"], "fast")
        self.assertEqual(fast_data["metadata"]["context"]["fields"], CONTEXT_FIELDS)
        self.assertIn("context", fast_data["metadata"]["timing_seconds"])
        self.assertIsInstance(fast_data["hbonds"][0]["env_h_sasa_fraction"], float)
        self.assertNotEqual(fast_rows[0]["env_h_sasa_fraction"], "")

        self.assertEqual(none_data["metadata"]["context_mode"], "none")
        self.assertIsNone(none_data["hbonds"][0]["env_h_sasa_fraction"])
        self.assertEqual(none_rows[0]["env_h_sasa_fraction"], "")

    def test_worker_metadata_records_requested_and_effective_counts(self) -> None:
        result = score_pdb(
            write_temp_pdb(two_hbond_lines()),
            hydrogen_mode="explicit",
            atom_types="ALL",
            context_mode="none",
            workers=100,
        )
        data = result.to_json_dict()

        self.assertEqual(data["metadata"]["workers"]["requested"], 100)
        self.assertGreaterEqual(data["metadata"]["workers"]["score"], 1)
        self.assertLessEqual(data["metadata"]["workers"]["score"], result.counts["donors"])
        self.assertEqual(data["metadata"]["workers"]["context"], 0)
        self.assertEqual(data["metadata"]["distance_cutoff"], 3.5)

    def test_custom_hbond_distance_cutoff_is_written_to_metadata(self) -> None:
        result = score_pdb(
            write_temp_pdb(two_hbond_lines()),
            hydrogen_mode="explicit",
            atom_types="ALL",
            context_mode="none",
            distance_cutoff=4.5,
        )

        self.assertEqual(result.to_json_dict()["metadata"]["distance_cutoff"], 4.5)

    def test_hbond_per_donor_hydrogen_metadata_records_raw_and_final_counts(self) -> None:
        result = score_pdb(
            write_temp_pdb(competing_acceptor_lines()),
            hydrogen_mode="explicit",
            context_mode="none",
            hbond_per_donor_hydrogen="best-distance",
            hydrogen_minimize="none",
        )
        metadata = result.to_json_dict()["metadata"]

        self.assertEqual(metadata["hbond_per_donor_hydrogen"], "best-distance")
        self.assertEqual(metadata["raw_hbond_candidates"], 2)
        self.assertEqual(metadata["counts"]["raw_hbonds"], 2)
        self.assertEqual(metadata["counts"]["hbonds"], 1)
        self.assertEqual(metadata["hydrogen_minimization"]["mode"], "none")
        self.assertFalse(metadata["hydrogen_minimization"]["ran"])

    def test_cli_workers_flag_writes_matching_outputs(self) -> None:
        input_path = write_temp_pdb(two_hbond_lines())
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "hbonds.json"
            csv_path = Path(tmpdir) / "hbonds.csv"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "score",
                        str(input_path),
                        "--hydrogen-mode",
                        "explicit",
                        "--atom-types",
                        "ALL",
                        "--workers",
                        "2",
                        "--hbond-distance-cutoff",
                        "4.5",
                        "--json",
                        str(json_path),
                        "--csv",
                        str(csv_path),
                    ]
                )

            data = json.loads(json_path.read_text())
            with csv_path.open() as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(exit_code, 0)
        self.assertEqual(data["metadata"]["workers"]["requested"], 2)
        self.assertEqual(data["metadata"]["distance_cutoff"], 4.5)
        self.assertEqual(len(data["hbonds"]), len(rows))
        self.assertEqual(len(data["hbonds"]), 2)

    def test_cli_hbond_reduction_flag_writes_matching_outputs_and_metadata(self) -> None:
        input_path = write_temp_pdb(competing_acceptor_lines())
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "hbonds.json"
            csv_path = Path(tmpdir) / "hbonds.csv"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "score",
                        str(input_path),
                        "--hydrogen-mode",
                        "explicit",
                        "--hydrogen-minimize",
                        "none",
                        "--hydrogen-forcefield",
                        "charmm",
                        "--ccd-online",
                        "never",
                        "--hbond-per-donor-hydrogen",
                        "best-distance",
                        "--context-mode",
                        "none",
                        "--json",
                        str(json_path),
                        "--csv",
                        str(csv_path),
                    ]
                )

            data = json.loads(json_path.read_text())
            with csv_path.open() as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(exit_code, 0)
        self.assertEqual(data["metadata"]["hbond_per_donor_hydrogen"], "best-distance")
        self.assertEqual(data["metadata"]["hydrogen_minimization"]["requested_forcefield"], "charmm")
        self.assertEqual(data["metadata"]["hydrogen_minimization"]["ccd_online"], "never")
        self.assertEqual(data["metadata"]["counts"]["raw_hbonds"], 2)
        self.assertEqual(len(data["hbonds"]), 1)
        self.assertEqual(len(rows), 1)

    def test_cli_rejects_invalid_workers(self) -> None:
        stderr = io.StringIO()
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(stderr):
                build_parser().parse_args(
                    [
                        "score",
                        "input.pdb",
                        "--workers",
                        "0",
                        "--json",
                        "out.json",
                        "--csv",
                        "out.csv",
                    ]
                )

        self.assertIn("positive integer", stderr.getvalue())

    def test_cli_rejects_invalid_hbond_distance_cutoff(self) -> None:
        stderr = io.StringIO()
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(stderr):
                build_parser().parse_args(
                    [
                        "score",
                        "input.pdb",
                        "--hbond-distance-cutoff",
                        "0",
                        "--json",
                        "out.json",
                        "--csv",
                        "out.csv",
                    ]
                )

        self.assertIn("positive number", stderr.getvalue())

    def test_cli_rejects_invalid_hbond_reduction_mode(self) -> None:
        stderr = io.StringIO()
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(stderr):
                build_parser().parse_args(
                    [
                        "score",
                        "input.pdb",
                        "--hbond-per-donor-hydrogen",
                        "nearest-ish",
                        "--json",
                        "out.json",
                        "--csv",
                        "out.csv",
                    ]
                )

        self.assertIn("invalid choice", stderr.getvalue())

    def test_rejects_invalid_python_api_modes(self) -> None:
        structure = parse_pdb(write_temp_pdb(simple_hbond_lines()))

        with self.assertRaisesRegex(ValueError, "hbond_per_donor_hydrogen"):
            score_structure(
                structure,
                context_mode="none",
                hbond_per_donor_hydrogen="nearest-ish",
            )
        with self.assertRaisesRegex(ValueError, "hydrogen_minimize"):
            score_pdb(
                write_temp_pdb(simple_hbond_lines()),
                hydrogen_mode="explicit",
                hydrogen_minimize="aggressive-ish",
            )
        with self.assertRaisesRegex(ValueError, "hydrogen_forcefield"):
            score_pdb(
                write_temp_pdb(simple_hbond_lines()),
                hydrogen_mode="explicit",
                hydrogen_forcefield="martini-ish",
            )
        with self.assertRaisesRegex(ValueError, "ccd_online"):
            score_pdb(
                write_temp_pdb(simple_hbond_lines()),
                hydrogen_mode="explicit",
                ccd_online="sometimes-ish",
            )

    def test_charmm_template_variants_add_one_set_per_modified_residue(self) -> None:
        residue = FakeOpenMMResidue(
            "TPO",
            [
                FakeOpenMMAtom("N", "N"),
                FakeOpenMMAtom("CA", "C"),
                FakeOpenMMAtom("CB", "C"),
                FakeOpenMMAtom("OG1", "O"),
                FakeOpenMMAtom("P", "P"),
                FakeOpenMMAtom("O1P", "O"),
                FakeOpenMMAtom("O2P", "O"),
                FakeOpenMMAtom("O3P", "O"),
                FakeOpenMMAtom("CG2", "C"),
                FakeOpenMMAtom("C", "C"),
                FakeOpenMMAtom("O", "O"),
            ],
            index=4,
        )
        template = FakeOpenMMTemplate(
            [
                FakeOpenMMAtom("N", "N"),
                FakeOpenMMAtom("HN", "H"),
                FakeOpenMMAtom("CA", "C"),
                FakeOpenMMAtom("HA", "H"),
                FakeOpenMMAtom("CB", "C"),
                FakeOpenMMAtom("HB", "H"),
                FakeOpenMMAtom("OG1", "O"),
                FakeOpenMMAtom("P", "P"),
                FakeOpenMMAtom("O1P", "O"),
                FakeOpenMMAtom("O2P", "O"),
                FakeOpenMMAtom("O3P", "O"),
                FakeOpenMMAtom("H3T", "H"),
                FakeOpenMMAtom("CG2", "C"),
                FakeOpenMMAtom("HG21", "H"),
                FakeOpenMMAtom("HG22", "H"),
                FakeOpenMMAtom("HG23", "H"),
                FakeOpenMMAtom("C", "C"),
                FakeOpenMMAtom("O", "O"),
            ],
            [
                (0, 1),
                (2, 3),
                (4, 5),
                (10, 11),
                (12, 13),
                (12, 14),
                (12, 15),
            ],
        )
        forcefield = SimpleNamespace(_templates={"TPO": template})

        variants, hydrogens_added = _charmm_template_hydrogen_variants(
            FakeOpenMMTopology([residue]),
            forcefield,
        )

        self.assertEqual(
            variants,
            [
                [
                    ("HN", "N"),
                    ("HA", "CA"),
                    ("HB", "CB"),
                    ("H3T", "O3P"),
                    ("HG21", "CG2"),
                    ("HG22", "CG2"),
                    ("HG23", "CG2"),
                ]
            ],
        )
        self.assertEqual(hydrogens_added[0]["name"], "TPO")
        self.assertEqual(hydrogens_added[0]["hydrogens"][0], {"name": "HN", "parent": "N"})

    def test_ccd_lookup_uses_cache_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cached = cache_dir / "TPO.cif"
            cached.write_text("data_TPO\n")

            lookup = _lookup_ccd_component("TPO", cache_dir=cache_dir, ccd_online="never")

        self.assertTrue(lookup["available"])
        self.assertEqual(lookup["source"], "cache")
        self.assertEqual(lookup["code"], "TPO")

    def test_add_hydrogens_can_skip_minimization_with_mocked_dependencies(self) -> None:
        class FakeFixer:
            def __init__(self, filename: str) -> None:
                self.filename = filename
                self.topology = object()
                self.positions = [(0.0, 0.0, 0.0)]
                self.ph = None

            def addMissingHydrogens(self, ph: float) -> None:
                self.ph = ph

        class FakePDBFile:
            @staticmethod
            def writeFile(topology, positions, handle, keepIds=True) -> None:
                handle.write("PDB\\n")

        app_module = module(
            "openmm.app",
            ForceField=object,
            HBonds=object(),
            Modeller=object,
            NoCutoff=object(),
            PDBFile=FakePDBFile,
            Simulation=object,
        )
        openmm_module = module(
            "openmm",
            __version__="8.test",
            CustomExternalForce=object,
            LocalEnergyMinimizer=object,
            Platform=object,
            VerletIntegrator=object,
            unit=SimpleNamespace(picoseconds=1.0, nanometer=1.0),
            app=app_module,
        )
        pdbfixer_module = module("pdbfixer", __version__="1.test", PDBFixer=FakeFixer)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "hydrated.pdb"
            with mock.patch.dict(
                sys.modules,
                {
                    "openmm": openmm_module,
                    "openmm.app": app_module,
                    "pdbfixer": pdbfixer_module,
                },
            ):
                metadata = add_hydrogens_with_pdbfixer(
                    "input.pdb",
                    output_path,
                    ph=6.5,
                    minimize="none",
                )

            self.assertEqual(output_path.read_text(), "PDB\\n")

        self.assertEqual(metadata["mode"], "none")
        self.assertFalse(metadata["ran"])
        self.assertEqual(metadata["openmm_version"], "8.test")
        self.assertEqual(metadata["pdbfixer_version"], "1.test")

    def test_add_hydrogens_runs_restrained_minimization_with_mocked_dependencies(self) -> None:
        class FakeElement:
            def __init__(self, symbol: str) -> None:
                self.symbol = symbol

        class FakeAtom:
            def __init__(self, symbol: str) -> None:
                self.element = FakeElement(symbol)

        class FakeTopology:
            def atoms(self):
                return iter([FakeAtom("N"), FakeAtom("H")])

        class FakeFixer:
            def __init__(self, filename: str) -> None:
                self.filename = filename
                self.topology = FakeTopology()
                self.positions = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0)]

            def addMissingHydrogens(self, ph: float) -> None:
                self.ph = ph

        class FakeForce:
            last = None

            def __init__(self, expression: str) -> None:
                self.expression = expression
                self.particles = []
                FakeForce.last = self

            def addGlobalParameter(self, name: str, value: float) -> None:
                self.global_parameter = (name, value)

            def addPerParticleParameter(self, name: str) -> None:
                pass

            def addParticle(self, index: int, parameters: list[float]) -> None:
                self.particles.append((index, parameters))

        class FakeSystem:
            def addForce(self, force) -> None:
                self.force = force

        class FakeForceField:
            def __init__(self, *files: str) -> None:
                self.files = files

            def createSystem(self, topology, nonbondedMethod, constraints):
                return FakeSystem()

        class FakePlatform:
            def __init__(self, name: str) -> None:
                self.name = name

            def getName(self) -> str:
                return self.name

        class FakePlatformFactory:
            @staticmethod
            def getPlatformByName(name: str):
                if name != "CPU":
                    raise RuntimeError(name)
                return FakePlatform(name)

        class FakeState:
            def getPositions(self):
                return [(0.0, 0.0, 0.0), (0.11, 0.0, 0.0)]

        class FakeContext:
            def setPositions(self, positions) -> None:
                self.positions = positions

            def getState(self, getPositions: bool):
                return FakeState()

        class FakeSimulation:
            def __init__(self, topology, system, integrator, platform=None) -> None:
                self.context = FakeContext()
                self.platform = platform

        class FakeMinimizer:
            called_with = None

            @staticmethod
            def minimize(context, tolerance: float, maxIterations: int) -> None:
                FakeMinimizer.called_with = (context, tolerance, maxIterations)

        class FakePDBFile:
            @staticmethod
            def writeFile(topology, positions, handle, keepIds=True) -> None:
                handle.write(str(positions))

        app_module = module(
            "openmm.app",
            ForceField=FakeForceField,
            HBonds="HBonds",
            Modeller=object,
            NoCutoff="NoCutoff",
            PDBFile=FakePDBFile,
            Simulation=FakeSimulation,
        )
        openmm_module = module(
            "openmm",
            __version__="8.test",
            CustomExternalForce=FakeForce,
            LocalEnergyMinimizer=FakeMinimizer,
            Platform=FakePlatformFactory,
            VerletIntegrator=lambda step: ("integrator", step),
            unit=SimpleNamespace(picoseconds=1.0, nanometer=1.0),
            app=app_module,
        )
        pdbfixer_module = module("pdbfixer", __version__="1.test", PDBFixer=FakeFixer)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "hydrated.pdb"
            with mock.patch.dict(
                sys.modules,
                {
                    "openmm": openmm_module,
                    "openmm.app": app_module,
                    "pdbfixer": pdbfixer_module,
                },
            ):
                metadata = add_hydrogens_with_pdbfixer(
                    "input.pdb",
                    output_path,
                    minimize="restrained",
                )

            self.assertIn("0.11", output_path.read_text())

        self.assertTrue(metadata["ran"])
        self.assertEqual(metadata["platform"], "CPU")
        self.assertEqual(metadata["selected_forcefield"], "amber")
        self.assertEqual(metadata["attempted_forcefields"], ["amber"])
        self.assertEqual(metadata["max_iterations"], 50)
        self.assertEqual(FakeForce.last.particles, [(0, [0.0, 0.0, 0.0])])
        self.assertEqual(FakeMinimizer.called_with[2], 50)

    def test_explicit_scoring_smoke_stays_fast_for_sample_pdb(self) -> None:
        start = time.perf_counter()
        result = score_pdb(Path("test_data/2erk.pdb"), hydrogen_mode="explicit", atom_types="ALL")
        elapsed = time.perf_counter() - start

        self.assertIn("donors", result.counts)
        self.assertIn("acceptors", result.counts)
        self.assertLess(elapsed, 1.0)

    def test_fast_context_smoke_stays_under_one_second(self) -> None:
        path = write_temp_pdb(simple_hbond_lines(h_shell_atoms()))

        start = time.perf_counter()
        result = score_pdb(path, hydrogen_mode="explicit", atom_types="ALL", context_mode="fast")
        elapsed = time.perf_counter() - start

        self.assertEqual(result.counts["hbonds"], 1)
        self.assertIsNotNone(result.hbonds[0].environment_context)
        self.assertLess(elapsed, 1.0)


if __name__ == "__main__":
    unittest.main()
