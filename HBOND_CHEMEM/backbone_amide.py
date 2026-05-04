"""Identify and score backbone amide HBonds in protein PDB files."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Iterable, Mapping, Sequence

from .pdb_io import Atom, ProteinStructure, find_backbone_acceptors, find_backbone_donors, parse_pdb
from .hydrogen import add_hydrogens_with_pdbfixer
from .reference_hbond_score import eval_poly, load_tables
from .score_bounds import NORMALIZATION_MODE, load_score_bounds, normalize_hbond_score


PEPTIDE_N_TYPE = 40
PEPTIDE_O_TYPE = 39
DISTANCE_CUTOFF = 6.0
ANGLE_CUTOFF = 110.0

CSV_FIELDS = [
    "model_id",
    "donor_atom_id",
    "donor_atom_name",
    "donor_chain_id",
    "donor_res_name",
    "donor_res_seq",
    "donor_ins_code",
    "donor_x",
    "donor_y",
    "donor_z",
    "donor_h_atom_id",
    "donor_h_atom_name",
    "donor_h_generated",
    "hydrogen_x",
    "hydrogen_y",
    "hydrogen_z",
    "acceptor_atom_id",
    "acceptor_atom_name",
    "acceptor_chain_id",
    "acceptor_res_name",
    "acceptor_res_seq",
    "acceptor_ins_code",
    "acceptor_x",
    "acceptor_y",
    "acceptor_z",
    "donor_acceptor_distance",
    "hydrogen_acceptor_distance",
    "dha_angle",
    "donor_type",
    "acceptor_type",
    "a_value",
    "b_value",
    "c_value",
    "hbond_score",
    "normalized_score",
]


@dataclass(frozen=True)
class BackboneHBond:
    """A scored backbone donor-H-acceptor interaction."""

    model_id: int
    donor: Atom
    hydrogen: Atom
    acceptor: Atom
    donor_acceptor_distance: float
    hydrogen_acceptor_distance: float
    dha_angle: float
    a_value: float
    b_value: float
    c_value: float
    hbond_score: float
    normalized_score: float

    def to_row(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "donor_atom_id": self.donor.atom_id,
            "donor_atom_name": self.donor.name,
            "donor_chain_id": self.donor.chain_id,
            "donor_res_name": self.donor.res_name,
            "donor_res_seq": self.donor.res_seq,
            "donor_ins_code": self.donor.ins_code,
            "donor_x": self.donor.x,
            "donor_y": self.donor.y,
            "donor_z": self.donor.z,
            "donor_h_atom_id": self.hydrogen.atom_id,
            "donor_h_atom_name": self.hydrogen.name,
            "donor_h_generated": self.hydrogen.generated,
            "hydrogen_x": self.hydrogen.x,
            "hydrogen_y": self.hydrogen.y,
            "hydrogen_z": self.hydrogen.z,
            "acceptor_atom_id": self.acceptor.atom_id,
            "acceptor_atom_name": self.acceptor.name,
            "acceptor_chain_id": self.acceptor.chain_id,
            "acceptor_res_name": self.acceptor.res_name,
            "acceptor_res_seq": self.acceptor.res_seq,
            "acceptor_ins_code": self.acceptor.ins_code,
            "acceptor_x": self.acceptor.x,
            "acceptor_y": self.acceptor.y,
            "acceptor_z": self.acceptor.z,
            "donor_acceptor_distance": self.donor_acceptor_distance,
            "hydrogen_acceptor_distance": self.hydrogen_acceptor_distance,
            "dha_angle": self.dha_angle,
            "donor_type": PEPTIDE_N_TYPE,
            "acceptor_type": PEPTIDE_O_TYPE,
            "a_value": self.a_value,
            "b_value": self.b_value,
            "c_value": self.c_value,
            "hbond_score": self.hbond_score,
            "normalized_score": self.normalized_score,
        }


@dataclass(frozen=True)
class ScoreResult:
    input_path: str
    hydrogen_mode: str
    hydrogen_source: str
    hbonds: list[BackboneHBond]
    counts: dict[str, int]
    timing_seconds: dict[str, float]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "metadata": {
                "input_path": self.input_path,
                "hydrogen_mode": self.hydrogen_mode,
                "hydrogen_source": self.hydrogen_source,
                "distance_cutoff": DISTANCE_CUTOFF,
                "angle_cutoff": ANGLE_CUTOFF,
                "donor_type": PEPTIDE_N_TYPE,
                "acceptor_type": PEPTIDE_O_TYPE,
                "normalization_mode": NORMALIZATION_MODE,
                "rep_cap_removed": True,
                "counts": self.counts,
                "timing_seconds": self.timing_seconds,
            },
            "hbonds": [hbond.to_row() for hbond in self.hbonds],
        }


def score_pdb(
    input_path: str | Path,
    *,
    hydrogen_mode: str = "auto",
    ph: float = 7.0,
) -> ScoreResult:
    """Load a PDB file, optionally add hydrogens, and score backbone amide HBonds."""

    if hydrogen_mode not in {"auto", "explicit", "pdbfixer"}:
        raise ValueError("hydrogen_mode must be one of: auto, explicit, pdbfixer")

    input_path = Path(input_path)
    total_start = time.perf_counter()
    parse_start = time.perf_counter()
    structure = parse_pdb(input_path)
    parse_seconds = time.perf_counter() - parse_start

    hydrogen_source = "explicit"
    hydrogen_seconds = 0.0
    if hydrogen_mode == "pdbfixer" or (
        hydrogen_mode == "auto" and not find_backbone_donors(structure)
    ):
        hydrogen_source = "pdbfixer"
        hydrogen_start = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmpdir:
            hydrated_path = Path(tmpdir) / "hydrated.pdb"
            add_hydrogens_with_pdbfixer(input_path, hydrated_path, ph=ph)
            hydrated = parse_pdb(hydrated_path)
        structure = hydrated.with_original_heavy_atom_ids(structure)
        hydrogen_seconds = time.perf_counter() - hydrogen_start

    score_start = time.perf_counter()
    hbonds = score_structure(structure)
    score_seconds = time.perf_counter() - score_start

    donors = find_backbone_donors(structure)
    acceptors = find_backbone_acceptors(structure)
    timing = {
        "parse": parse_seconds,
        "hydrogen": hydrogen_seconds,
        "score": score_seconds,
        "total": time.perf_counter() - total_start,
    }
    counts = {
        "atoms": len(structure.atoms),
        "backbone_donors": len(donors),
        "backbone_acceptors": len(acceptors),
        "hbonds": len(hbonds),
    }
    return ScoreResult(
        input_path=str(input_path),
        hydrogen_mode=hydrogen_mode,
        hydrogen_source=hydrogen_source,
        hbonds=hbonds,
        counts=counts,
        timing_seconds=timing,
    )


def score_structure(
    structure: ProteinStructure,
    *,
    distance_cutoff: float = DISTANCE_CUTOFF,
    angle_cutoff: float = ANGLE_CUTOFF,
) -> list[BackboneHBond]:
    """Score all backbone N-H to backbone O pairs that pass ChemEM cutoffs."""

    tables = load_tables()
    coeff_a = tables.poly_a[str(PEPTIDE_N_TYPE)][str(PEPTIDE_O_TYPE)]
    coeff_b = tables.poly_b[str(PEPTIDE_N_TYPE)][str(PEPTIDE_O_TYPE)]
    coeff_c = tables.poly_c[str(PEPTIDE_N_TYPE)][str(PEPTIDE_O_TYPE)]
    score_bounds = load_score_bounds()

    donors = find_backbone_donors(structure)
    acceptors = find_backbone_acceptors(structure)
    acceptor_grid = _build_spatial_grid(acceptors, distance_cutoff)
    cutoff2 = distance_cutoff * distance_cutoff

    hbonds: list[BackboneHBond] = []
    for donor in donors:
        for acceptor in _nearby_acceptors(donor.atom, acceptor_grid, distance_cutoff):
            donor_acceptor_distance2 = squared_distance(donor.atom.xyz, acceptor.xyz)
            if donor_acceptor_distance2 >= cutoff2:
                continue

            best_hydrogen = None
            best_angle = -1.0
            for hydrogen in donor.hydrogens:
                angle = calc_bond_angle(donor.atom.xyz, hydrogen.xyz, acceptor.xyz)
                if angle > best_angle:
                    best_angle = angle
                    best_hydrogen = hydrogen

            if best_hydrogen is None or best_angle <= angle_cutoff:
                continue

            donor_acceptor_distance = math.sqrt(donor_acceptor_distance2)
            a_value = eval_poly(coeff_a, best_angle)
            b_value = eval_poly(coeff_b, best_angle)
            c_value = eval_poly(coeff_c, best_angle)
            hbond_score = chemem_hbond_score(
                a_value,
                b_value,
                c_value,
                donor_acceptor_distance,
            )
            normalized_score = normalize_hbond_score(
                hbond_score,
                PEPTIDE_N_TYPE,
                PEPTIDE_O_TYPE,
                score_bounds,
            )
            hbonds.append(
                BackboneHBond(
                    model_id=donor.atom.model_id,
                    donor=donor.atom,
                    hydrogen=best_hydrogen,
                    acceptor=acceptor,
                    donor_acceptor_distance=donor_acceptor_distance,
                    hydrogen_acceptor_distance=distance(best_hydrogen.xyz, acceptor.xyz),
                    dha_angle=best_angle,
                    a_value=a_value,
                    b_value=b_value,
                    c_value=c_value,
                    hbond_score=hbond_score,
                    normalized_score=normalized_score,
                )
            )

    hbonds.sort(
        key=lambda item: (
            item.model_id,
            item.donor.chain_id,
            item.donor.res_seq,
            item.acceptor.chain_id,
            item.acceptor.res_seq,
            str(item.acceptor.serial),
        )
    )
    return hbonds


def chemem_hbond_score(
    a_value: float,
    b_value: float,
    c_value: float,
    distance_angstrom: float,
) -> float:
    """Evaluate the ChemEM HBond Buckingham-style term without repCap."""

    r_clamped = max(distance_angstrom, 2.0)
    r2 = r_clamped * r_clamped
    r6 = r2 * r2 * r2
    return a_value * math.exp(-b_value * r_clamped) - c_value / r6


def write_json(result: ScoreResult, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(result.to_json_dict(), indent=2) + "\n")


def write_csv(result: ScoreResult, output_path: str | Path) -> None:
    with Path(output_path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for hbond in result.hbonds:
            writer.writerow(hbond.to_row())


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(squared_distance(a, b))


def squared_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return dx * dx + dy * dy + dz * dz


def calc_bond_angle(
    p1: Sequence[float],
    p2: Sequence[float],
    p3: Sequence[float],
) -> float:
    """Angle in degrees at p2 between p1-p2-p3."""

    ba = (float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]), float(p1[2]) - float(p2[2]))
    bc = (float(p3[0]) - float(p2[0]), float(p3[1]) - float(p2[1]), float(p3[2]) - float(p2[2]))
    ba_norm = math.sqrt(ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2])
    bc_norm = math.sqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2])
    if ba_norm == 0.0 or bc_norm == 0.0:
        return -1.0

    cosang = (ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]) / (ba_norm * bc_norm)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def _build_spatial_grid(atoms: Iterable[Atom], cell_size: float) -> dict[tuple[int, int, int], list[Atom]]:
    grid: dict[tuple[int, int, int], list[Atom]] = defaultdict(list)
    for atom in atoms:
        grid[_grid_key(atom.xyz, cell_size)].append(atom)
    return grid


def _nearby_acceptors(
    donor: Atom,
    grid: Mapping[tuple[int, int, int], list[Atom]],
    cell_size: float,
) -> Iterable[Atom]:
    cx, cy, cz = _grid_key(donor.xyz, cell_size)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield from grid.get((cx + dx, cy + dy, cz + dz), ())


def _grid_key(coord: Sequence[float], cell_size: float) -> tuple[int, int, int]:
    return (
        math.floor(float(coord[0]) / cell_size),
        math.floor(float(coord[1]) / cell_size),
        math.floor(float(coord[2]) / cell_size),
    )
