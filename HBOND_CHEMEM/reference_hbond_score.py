#!/usr/bin/env python3
"""Reference implementation of the ChemEM HBond contribution.

This mirrors the HBond branch in ``ChemEM/cpp/docking/echo_score.cpp``
(``echo_score_v2``) without importing ChemEM or RDKit.  Coordinates are
expected in Angstrom.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

DONOR_BIT = 1
ACCEPTOR_BIT = 2


@dataclass(frozen=True)
class HBondTables:
    donor_ids_from_data_py: set[int]
    acceptor_ids_from_data_py: set[int]
    polynomial_donor_ids: set[int]
    polynomial_acceptor_ids: set[int]
    poly_a: dict[str, dict[str, list[float]]]
    poly_b: dict[str, dict[str, list[float]]]
    poly_c: dict[str, dict[str, list[float]]]


@dataclass(frozen=True)
class HBondScore:
    contribution: float
    raw_value: float
    distance: float
    best_angle: float
    donor_type: int | None
    acceptor_type: int | None
    reason: str


def load_tables(data_dir: Path = DATA_DIR) -> HBondTables:
    roles = json.loads((data_dir / "hbond_roles.json").read_text())
    poly_a = json.loads((data_dir / "HBOND_POLY_A.json").read_text())
    poly_b = json.loads((data_dir / "HBOND_POLY_B.json").read_text())
    poly_c = json.loads((data_dir / "HBOND_POLY_C.json").read_text())

    return HBondTables(
        donor_ids_from_data_py=set(roles["hbond_donor_atom_type_ids_from_data_py"]),
        acceptor_ids_from_data_py=set(roles["hbond_acceptor_atom_type_ids_from_data_py"]),
        polynomial_donor_ids=set(roles["hbond_polynomial_donor_type_ids"]),
        polynomial_acceptor_ids=set(roles["hbond_polynomial_acceptor_type_ids"]),
        poly_a=poly_a,
        poly_b=poly_b,
        poly_c=poly_c,
    )


def role_from_ids(atom_type: int, donor_ids: set[int], acceptor_ids: set[int]) -> int:
    role = 0
    if atom_type in donor_ids:
        role |= DONOR_BIT
    if atom_type in acceptor_ids:
        role |= ACCEPTOR_BIT
    return role


def mask_allows_hbond(
    protein_atom_type: int,
    ligand_atom_type: int,
    tables: HBondTables,
) -> bool:
    """Match ``compute_donor_acceptor_mask`` from precompute_data.py."""

    protein_role = role_from_ids(
        protein_atom_type,
        tables.donor_ids_from_data_py,
        tables.acceptor_ids_from_data_py,
    )
    ligand_role = role_from_ids(
        ligand_atom_type,
        tables.donor_ids_from_data_py,
        tables.acceptor_ids_from_data_py,
    )
    return bool(
        (protein_role & DONOR_BIT and ligand_role & ACCEPTOR_BIT)
        or (protein_role & ACCEPTOR_BIT and ligand_role & DONOR_BIT)
    )


def polynomial_role(atom_type: int, tables: HBondTables) -> int:
    """Match C++ ``PrecomputedDataCPP2::get_hbond_role``."""

    return role_from_ids(
        atom_type,
        tables.polynomial_donor_ids,
        tables.polynomial_acceptor_ids,
    )


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def calc_bond_angle(
    p1: Sequence[float],
    p2: Sequence[float],
    p3: Sequence[float],
) -> float:
    """Angle in degrees at p2 between p1-p2-p3."""

    ba = [float(x) - float(y) for x, y in zip(p1, p2)]
    bc = [float(x) - float(y) for x, y in zip(p3, p2)]
    ba_norm = math.sqrt(sum(x * x for x in ba))
    bc_norm = math.sqrt(sum(x * x for x in bc))
    if ba_norm == 0.0 or bc_norm == 0.0:
        return -1.0

    cosang = sum(x * y for x, y in zip(ba, bc)) / (ba_norm * bc_norm)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def best_hbond_angle(
    donor_pos: Sequence[float],
    donor_h_positions: Iterable[Sequence[float]],
    acceptor_pos: Sequence[float],
) -> float:
    best = -1.0
    for h_pos in donor_h_positions:
        best = max(best, calc_bond_angle(donor_pos, h_pos, acceptor_pos))
    return best


def eval_poly(coefficients: Sequence[float], x: float) -> float:
    """Horner evaluation with the same coefficient order as C++ eval_poly."""

    if not coefficients:
        raise ValueError("Polynomial coefficient list is empty")
    result = float(coefficients[0])
    for coeff in coefficients[1:]:
        result = result * x + float(coeff)
    return result


def cap_positive(value: float, rep_cap: float = 5.0) -> float:
    if value <= 0.0:
        return value
    return rep_cap * math.tanh(value / rep_cap)


def score_directed_hbond(
    donor_type: int,
    acceptor_type: int,
    donor_pos: Sequence[float],
    donor_h_positions: Iterable[Sequence[float]],
    acceptor_pos: Sequence[float],
    *,
    tables: HBondTables | None = None,
    env_scale: float = 1.0,
    rep_cap: float = 5.0,
    distance_cutoff: float = 6.0,
    angle_cutoff: float = 110.0,
) -> HBondScore:
    """Score a known donor -> acceptor direction."""

    tables = tables or load_tables()
    r = distance(donor_pos, acceptor_pos)
    if r >= distance_cutoff:
        return HBondScore(0.0, 0.0, r, -1.0, donor_type, acceptor_type, "distance_cutoff")

    angle = best_hbond_angle(donor_pos, donor_h_positions, acceptor_pos)
    if angle <= angle_cutoff:
        return HBondScore(0.0, 0.0, r, angle, donor_type, acceptor_type, "angle_cutoff")

    donor_key = str(donor_type)
    acceptor_key = str(acceptor_type)
    try:
        coeff_a = tables.poly_a[donor_key][acceptor_key]
        coeff_b = tables.poly_b[donor_key][acceptor_key]
        coeff_c = tables.poly_c[donor_key][acceptor_key]
    except KeyError:
        return HBondScore(0.0, 0.0, r, angle, donor_type, acceptor_type, "missing_polynomial")

    a_value = eval_poly(coeff_a, angle)
    b_value = eval_poly(coeff_b, angle)
    c_value = eval_poly(coeff_c, angle)

    r_clamped = max(r, 2.0)
    r2 = r_clamped * r_clamped
    r6 = r2 * r2 * r2
    raw = a_value * math.exp(-b_value * r_clamped) - c_value / r6
    if raw < 0.0:
        raw *= env_scale

    return HBondScore(
        contribution=cap_positive(raw, rep_cap),
        raw_value=raw,
        distance=r,
        best_angle=angle,
        donor_type=donor_type,
        acceptor_type=acceptor_type,
        reason="scored",
    )


def score_hbond_pair(
    ligand_atom_type: int,
    protein_atom_type: int,
    ligand_pos: Sequence[float],
    protein_pos: Sequence[float],
    *,
    ligand_h_positions: Iterable[Sequence[float]] = (),
    protein_h_positions: Iterable[Sequence[float]] = (),
    protein_role: int | None = None,
    tables: HBondTables | None = None,
    env_scale: float = 1.0,
    rep_cap: float = 5.0,
    distance_cutoff: float = 6.0,
    angle_cutoff: float = 110.0,
) -> HBondScore:
    """Score one ligand/protein atom pair as ``echo_score_v2`` does.

    ``protein_role`` is the role integer from ``get_role_int``:
    donor=1, acceptor=2, both=3.  If omitted, it is inferred from atom type IDs,
    which is useful for isolated examples but less exact than the active protein
    residue/atom-name role path.
    """

    tables = tables or load_tables()
    r = distance(ligand_pos, protein_pos)
    if r >= distance_cutoff:
        return HBondScore(0.0, 0.0, r, -1.0, None, None, "distance_cutoff")
    if not mask_allows_hbond(protein_atom_type, ligand_atom_type, tables):
        return HBondScore(0.0, 0.0, r, -1.0, None, None, "mask_rejected")

    if protein_role is None:
        protein_role = role_from_ids(
            protein_atom_type,
            tables.donor_ids_from_data_py,
            tables.acceptor_ids_from_data_py,
        )
    ligand_role = polynomial_role(ligand_atom_type, tables)

    best_angle = -1.0
    donor_type = None
    acceptor_type = None
    donor_pos = None
    donor_h_positions = ()
    acceptor_pos = None

    if (protein_role & DONOR_BIT) and (ligand_role & ACCEPTOR_BIT):
        angle = best_hbond_angle(protein_pos, protein_h_positions, ligand_pos)
        if angle > best_angle:
            best_angle = angle
            donor_type = protein_atom_type
            acceptor_type = ligand_atom_type
            donor_pos = protein_pos
            donor_h_positions = protein_h_positions
            acceptor_pos = ligand_pos

    if (protein_role & ACCEPTOR_BIT) and (ligand_role & DONOR_BIT):
        angle = best_hbond_angle(ligand_pos, ligand_h_positions, protein_pos)
        if angle > best_angle:
            best_angle = angle
            donor_type = ligand_atom_type
            acceptor_type = protein_atom_type
            donor_pos = ligand_pos
            donor_h_positions = ligand_h_positions
            acceptor_pos = protein_pos

    if donor_type is None or acceptor_type is None:
        return HBondScore(0.0, 0.0, r, best_angle, None, None, "no_valid_direction")

    return score_directed_hbond(
        donor_type,
        acceptor_type,
        donor_pos,
        donor_h_positions,
        acceptor_pos,
        tables=tables,
        env_scale=env_scale,
        rep_cap=rep_cap,
        distance_cutoff=distance_cutoff,
        angle_cutoff=angle_cutoff,
    )


def _self_test() -> None:
    tables = load_tables()

    good = score_hbond_pair(
        ligand_atom_type=13,
        protein_atom_type=19,
        ligand_pos=(0.0, 0.0, 0.0),
        ligand_h_positions=[(1.0, 0.0, 0.0)],
        protein_pos=(2.8, 0.0, 0.0),
        protein_role=ACCEPTOR_BIT,
        tables=tables,
    )
    assert good.reason == "scored", good
    assert good.donor_type == 13
    assert good.acceptor_type == 19
    assert good.best_angle > 110.0

    bad_angle = score_hbond_pair(
        ligand_atom_type=13,
        protein_atom_type=19,
        ligand_pos=(0.0, 0.0, 0.0),
        ligand_h_positions=[(1.0, 0.0, 0.0)],
        protein_pos=(0.5, 1.0, 0.0),
        protein_role=ACCEPTOR_BIT,
        tables=tables,
    )
    assert bad_angle.reason == "angle_cutoff", bad_angle

    too_far = score_hbond_pair(
        ligand_atom_type=13,
        protein_atom_type=19,
        ligand_pos=(0.0, 0.0, 0.0),
        ligand_h_positions=[(1.0, 0.0, 0.0)],
        protein_pos=(6.0, 0.0, 0.0),
        protein_role=ACCEPTOR_BIT,
        tables=tables,
    )
    assert too_far.reason == "distance_cutoff", too_far

    capped = cap_positive(10.0, rep_cap=5.0)
    assert 0.0 < capped <= 5.0
    assert math.isclose(capped, 5.0 * math.tanh(10.0 / 5.0))
    print("HBond reference self-test passed")
    print(good)


if __name__ == "__main__":
    _self_test()
