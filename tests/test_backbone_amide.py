from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from HBOND_CHEMEM.backbone_amide import (
    PEPTIDE_N_TYPE,
    PEPTIDE_O_TYPE,
    chemem_hbond_score,
    score_pdb,
    score_structure,
    write_csv,
    write_json,
)
from HBOND_CHEMEM.pdb_io import find_backbone_donors, parse_pdb
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
        self.assertEqual(hbonds[0].acceptor.serial, 3)
        self.assertGreater(hbonds[0].dha_angle, 110.0)

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


if __name__ == "__main__":
    unittest.main()
