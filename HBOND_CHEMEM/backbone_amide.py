"""Identify and score protein HBonds in PDB files."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import csv
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Iterable, Mapping, Sequence

from .pdb_io import Atom, ProteinStructure, parse_pdb
from .environment_context import (
    CONTEXT_FIELDS,
    CONTEXT_MODE_FAST,
    CONTEXT_MODES,
    EnvironmentContext,
    FastEnvironmentContextCalculator,
    fast_context_metadata,
)
from .hydrogen import add_hydrogens_with_pdbfixer
from .hydrogen import (
    CCD_ONLINE_AUTO,
    HYDROGEN_FORCEFIELD_AUTO,
    HYDROGEN_MINIMIZE_AUTO,
    hydrogen_minimization_metadata,
    validate_ccd_online,
    validate_hydrogen_forcefield,
    validate_hydrogen_minimize,
)
from .protein_hbond_typing import (
    AtomSelector,
    PEPTIDE_N_TYPE,
    PEPTIDE_O_TYPE,
    ProteinAcceptor,
    ProteinDonor,
    find_protein_acceptors,
    find_protein_donors,
    parse_atom_selector,
)
from .reference_hbond_score import eval_poly, load_tables
from .score_bounds import NORMALIZATION_MODE, load_score_bounds, normalize_hbond_score


DISTANCE_CUTOFF = 3.5
ANGLE_CUTOFF = 110.0
HBOND_PER_DONOR_HYDROGEN_BEST_DISTANCE = "best-distance"
HBOND_PER_DONOR_HYDROGEN_BEST_NORMALIZED_SCORE = "best-normalized-score"
HBOND_PER_DONOR_HYDROGEN_MODES = {
    HBOND_PER_DONOR_HYDROGEN_BEST_DISTANCE,
    HBOND_PER_DONOR_HYDROGEN_BEST_NORMALIZED_SCORE,
}

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
] + CONTEXT_FIELDS


@dataclass(frozen=True)
class ProteinHBond:
    """A scored protein donor-H-acceptor interaction."""

    model_id: int
    donor: Atom
    hydrogen: Atom
    acceptor: Atom
    donor_type: int
    acceptor_type: int
    donor_acceptor_distance: float
    hydrogen_acceptor_distance: float
    dha_angle: float
    a_value: float
    b_value: float
    c_value: float
    hbond_score: float
    normalized_score: float
    environment_context: EnvironmentContext | None = None

    def to_row(self) -> dict[str, object]:
        row = {
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
            "donor_type": self.donor_type,
            "acceptor_type": self.acceptor_type,
            "a_value": self.a_value,
            "b_value": self.b_value,
            "c_value": self.c_value,
            "hbond_score": self.hbond_score,
            "normalized_score": self.normalized_score,
        }
        if self.environment_context is None:
            row.update({field: None for field in CONTEXT_FIELDS})
        else:
            row.update(self.environment_context.to_row())
        return row


BackboneHBond = ProteinHBond


@dataclass(frozen=True)
class ScoreResult:
    input_path: str
    hydrogen_mode: str
    hydrogen_source: str
    hydrogen_minimization: dict[str, object]
    atom_types: str
    hbond_per_donor_hydrogen: str | None
    context_mode: str
    context_metadata: dict[str, object]
    requested_workers: int
    effective_score_workers: int
    effective_context_workers: int
    distance_cutoff: float
    raw_hbond_count: int
    hbonds: list[ProteinHBond]
    counts: dict[str, int]
    timing_seconds: dict[str, float]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "metadata": {
                "input_path": self.input_path,
                "hydrogen_mode": self.hydrogen_mode,
                "hydrogen_source": self.hydrogen_source,
                "hydrogen_minimization": self.hydrogen_minimization,
                "atom_types": self.atom_types,
                "hbond_per_donor_hydrogen": self.hbond_per_donor_hydrogen,
                "raw_hbond_candidates": self.raw_hbond_count,
                "distance_cutoff": self.distance_cutoff,
                "angle_cutoff": ANGLE_CUTOFF,
                "donor_types": sorted({hbond.donor_type for hbond in self.hbonds}),
                "acceptor_types": sorted({hbond.acceptor_type for hbond in self.hbonds}),
                "normalization_mode": NORMALIZATION_MODE,
                "rep_cap_removed": True,
                "context_mode": self.context_mode,
                "context": self.context_metadata,
                "workers": {
                    "requested": self.requested_workers,
                    "score": self.effective_score_workers,
                    "context": self.effective_context_workers,
                },
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
    atom_types: str | Sequence[str] | None = "N",
    context_mode: str = CONTEXT_MODE_FAST,
    workers: int = 1,
    distance_cutoff: float = DISTANCE_CUTOFF,
    hbond_per_donor_hydrogen: str | None = None,
    hydrogen_minimize: str = HYDROGEN_MINIMIZE_AUTO,
    hydrogen_forcefield: str = HYDROGEN_FORCEFIELD_AUTO,
    ccd_cache: str | Path | None = None,
    ccd_online: str = CCD_ONLINE_AUTO,
) -> ScoreResult:
    """Load a PDB file, optionally add hydrogens, and score protein HBonds."""

    if hydrogen_mode not in {"auto", "explicit", "pdbfixer"}:
        raise ValueError("hydrogen_mode must be one of: auto, explicit, pdbfixer")
    _validate_context_mode(context_mode)
    requested_workers = _validate_workers(workers)
    distance_cutoff = _validate_positive_float(distance_cutoff, "distance_cutoff")
    hbond_per_donor_hydrogen = _validate_hbond_per_donor_hydrogen(hbond_per_donor_hydrogen)
    hydrogen_minimize = validate_hydrogen_minimize(hydrogen_minimize)
    hydrogen_forcefield = validate_hydrogen_forcefield(hydrogen_forcefield)
    ccd_online = validate_ccd_online(ccd_online)
    selector = parse_atom_selector(atom_types)

    input_path = Path(input_path)
    total_start = time.perf_counter()
    parse_start = time.perf_counter()
    structure = parse_pdb(input_path)
    parse_seconds = time.perf_counter() - parse_start

    hydrogen_source = "explicit"
    hydrogen_seconds = 0.0
    hydrogen_minimization = hydrogen_minimization_metadata(
        hydrogen_minimize,
        ran=False,
        requested_forcefield=hydrogen_forcefield,
        ccd_cache=str(ccd_cache) if ccd_cache is not None else None,
        ccd_online=ccd_online,
    )
    if hydrogen_mode == "pdbfixer" or (
        hydrogen_mode == "auto" and not _has_relevant_explicit_donors(structure, selector)
    ):
        hydrogen_source = "pdbfixer"
        hydrogen_start = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmpdir:
            hydrated_path = Path(tmpdir) / "hydrated.pdb"
            hydrogen_minimization = add_hydrogens_with_pdbfixer(
                input_path,
                hydrated_path,
                ph=ph,
                minimize=hydrogen_minimize,
                hydrogen_forcefield=hydrogen_forcefield,
                ccd_cache=ccd_cache,
                ccd_online=ccd_online,
            )
            hydrated = parse_pdb(hydrated_path)
        structure = hydrated.with_original_heavy_atom_ids(structure)
        hydrogen_seconds = time.perf_counter() - hydrogen_start

    score_start = time.perf_counter()
    hbonds, effective_score_workers, raw_hbond_count = _score_structure_base_with_worker_count(
        structure,
        atom_types=selector.raw,
        distance_cutoff=distance_cutoff,
        workers=requested_workers,
        hbond_per_donor_hydrogen=hbond_per_donor_hydrogen,
    )
    score_seconds = time.perf_counter() - score_start

    context_start = time.perf_counter()
    effective_context_workers = 0
    if context_mode == CONTEXT_MODE_FAST:
        hbonds, effective_context_workers = _with_environment_context_worker_count(
            structure,
            hbonds,
            workers=requested_workers,
        )
    context_seconds = time.perf_counter() - context_start

    donors = find_protein_donors(structure)
    acceptors = find_protein_acceptors(structure)
    timing = {
        "parse": parse_seconds,
        "hydrogen": hydrogen_seconds,
        "score": score_seconds,
        "context": context_seconds,
        "total": time.perf_counter() - total_start,
    }
    counts = {
        "atoms": len(structure.atoms),
        "donors": len(donors),
        "acceptors": len(acceptors),
        "selected_donors": sum(1 for donor in donors if selector.matches(donor.atom, donor.atom_type)),
        "selected_acceptors": sum(
            1 for acceptor in acceptors if selector.matches(acceptor.atom, acceptor.atom_type)
        ),
        "backbone_donors": sum(1 for donor in donors if donor.atom.name == "N"),
        "backbone_acceptors": sum(1 for acceptor in acceptors if acceptor.atom.name == "O"),
        "raw_hbonds": raw_hbond_count,
        "hbonds": len(hbonds),
    }
    return ScoreResult(
        input_path=str(input_path),
        hydrogen_mode=hydrogen_mode,
        hydrogen_source=hydrogen_source,
        hydrogen_minimization=hydrogen_minimization,
        atom_types=selector.raw,
        hbond_per_donor_hydrogen=hbond_per_donor_hydrogen,
        context_mode=context_mode,
        context_metadata=fast_context_metadata(context_mode),
        requested_workers=requested_workers,
        effective_score_workers=effective_score_workers,
        effective_context_workers=effective_context_workers,
        distance_cutoff=distance_cutoff,
        raw_hbond_count=raw_hbond_count,
        hbonds=hbonds,
        counts=counts,
        timing_seconds=timing,
    )


def score_structure(
    structure: ProteinStructure,
    *,
    atom_types: str | Sequence[str] | None = "N",
    distance_cutoff: float = DISTANCE_CUTOFF,
    angle_cutoff: float = ANGLE_CUTOFF,
    context_mode: str = CONTEXT_MODE_FAST,
    workers: int = 1,
    hbond_per_donor_hydrogen: str | None = None,
) -> list[ProteinHBond]:
    """Score protein HBonds where the selected atom participates."""

    _validate_context_mode(context_mode)
    requested_workers = _validate_workers(workers)
    distance_cutoff = _validate_positive_float(distance_cutoff, "distance_cutoff")
    hbonds = _score_structure_base(
        structure,
        atom_types=atom_types,
        distance_cutoff=distance_cutoff,
        angle_cutoff=angle_cutoff,
        workers=requested_workers,
        hbond_per_donor_hydrogen=hbond_per_donor_hydrogen,
    )
    if context_mode == CONTEXT_MODE_FAST:
        return _with_environment_context(structure, hbonds, workers=requested_workers)
    return hbonds


def _score_structure_base(
    structure: ProteinStructure,
    *,
    atom_types: str | Sequence[str] | None = "N",
    distance_cutoff: float = DISTANCE_CUTOFF,
    angle_cutoff: float = ANGLE_CUTOFF,
    workers: int = 1,
    hbond_per_donor_hydrogen: str | None = None,
) -> list[ProteinHBond]:
    """Score protein HBonds without adding environment context fields."""

    hbonds, _, _ = _score_structure_base_with_worker_count(
        structure,
        atom_types=atom_types,
        distance_cutoff=distance_cutoff,
        angle_cutoff=angle_cutoff,
        workers=workers,
        hbond_per_donor_hydrogen=hbond_per_donor_hydrogen,
    )
    return hbonds


def _score_structure_base_with_worker_count(
    structure: ProteinStructure,
    *,
    atom_types: str | Sequence[str] | None = "N",
    distance_cutoff: float = DISTANCE_CUTOFF,
    angle_cutoff: float = ANGLE_CUTOFF,
    workers: int = 1,
    hbond_per_donor_hydrogen: str | None = None,
) -> tuple[list[ProteinHBond], int, int]:
    """Score protein HBonds and report the effective worker count used."""

    distance_cutoff = _validate_positive_float(distance_cutoff, "distance_cutoff")
    hbond_per_donor_hydrogen = _validate_hbond_per_donor_hydrogen(hbond_per_donor_hydrogen)
    selector = parse_atom_selector(atom_types)
    tables = load_tables()
    score_bounds = load_score_bounds()

    donors = find_protein_donors(structure)
    acceptors = find_protein_acceptors(structure)
    acceptor_grid = _build_spatial_grid(acceptors, distance_cutoff)
    effective_workers = _effective_worker_count(workers, len(donors))

    if effective_workers == 1:
        hbonds = _score_donor_chunk(
            donors,
            selector,
            tables,
            score_bounds,
            acceptor_grid,
            distance_cutoff,
            angle_cutoff,
            hbond_per_donor_hydrogen,
        )
    else:
        hbonds = []
        chunks = _chunk_sequence(donors, effective_workers)
        tasks = [
            (
                chunk,
                selector,
                tables,
                score_bounds,
                acceptor_grid,
                distance_cutoff,
                angle_cutoff,
                hbond_per_donor_hydrogen,
            )
            for chunk in chunks
        ]
        try:
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                for chunk_hbonds in executor.map(_score_donor_chunk_from_args, tasks):
                    hbonds.extend(chunk_hbonds)
        except OSError:
            effective_workers = 1
            hbonds = _score_donor_chunk(
                donors,
                selector,
                tables,
                score_bounds,
                acceptor_grid,
                distance_cutoff,
                angle_cutoff,
                hbond_per_donor_hydrogen,
            )

    raw_hbond_count = len(hbonds)
    if hbond_per_donor_hydrogen is not None:
        hbonds = _select_one_hbond_per_donor_hydrogen(hbonds, hbond_per_donor_hydrogen)

    _sort_hbonds(hbonds)
    return hbonds, effective_workers, raw_hbond_count


def _score_donor_chunk_from_args(args: tuple[object, ...]) -> list[ProteinHBond]:
    return _score_donor_chunk(*args)


def _score_donor_chunk(
    donors: Sequence[ProteinDonor],
    selector: AtomSelector,
    tables,
    score_bounds: dict[str, object],
    acceptor_grid: Mapping[tuple[int, int, int], list[ProteinAcceptor]],
    distance_cutoff: float,
    angle_cutoff: float,
    hbond_per_donor_hydrogen: str | None,
) -> list[ProteinHBond]:
    cutoff2 = distance_cutoff * distance_cutoff
    hbonds: list[ProteinHBond] = []

    for donor in donors:
        for acceptor in _nearby_acceptors(donor.atom, acceptor_grid, distance_cutoff):
            if _same_atom(donor.atom, acceptor.atom):
                continue
            if not (
                selector.matches(donor.atom, donor.atom_type)
                or selector.matches(acceptor.atom, acceptor.atom_type)
            ):
                continue

            donor_key = str(donor.atom_type)
            acceptor_key = str(acceptor.atom_type)
            try:
                coeff_a = tables.poly_a[donor_key][acceptor_key]
                coeff_b = tables.poly_b[donor_key][acceptor_key]
                coeff_c = tables.poly_c[donor_key][acceptor_key]
            except KeyError:
                continue

            donor_acceptor_distance2 = squared_distance(donor.atom.xyz, acceptor.atom.xyz)
            if donor_acceptor_distance2 >= cutoff2:
                continue

            donor_acceptor_distance = math.sqrt(donor_acceptor_distance2)
            if hbond_per_donor_hydrogen is None:
                best_hydrogen = None
                best_angle = -1.0
                for hydrogen in donor.hydrogens:
                    angle = calc_bond_angle(donor.atom.xyz, hydrogen.xyz, acceptor.atom.xyz)
                    if angle > best_angle:
                        best_angle = angle
                        best_hydrogen = hydrogen

                if best_hydrogen is None or best_angle <= angle_cutoff:
                    continue
                hbonds.append(
                    _build_hbond(
                        donor,
                        best_hydrogen,
                        acceptor,
                        donor_acceptor_distance,
                        best_angle,
                        coeff_a,
                        coeff_b,
                        coeff_c,
                        score_bounds,
                    )
                )
                continue

            for hydrogen in donor.hydrogens:
                angle = calc_bond_angle(donor.atom.xyz, hydrogen.xyz, acceptor.atom.xyz)
                if angle <= angle_cutoff:
                    continue
                hbonds.append(
                    _build_hbond(
                        donor,
                        hydrogen,
                        acceptor,
                        donor_acceptor_distance,
                        angle,
                        coeff_a,
                        coeff_b,
                        coeff_c,
                        score_bounds,
                    )
                )
    return hbonds


def _build_hbond(
    donor: ProteinDonor,
    hydrogen: Atom,
    acceptor: ProteinAcceptor,
    donor_acceptor_distance: float,
    angle: float,
    coeff_a: Sequence[float],
    coeff_b: Sequence[float],
    coeff_c: Sequence[float],
    score_bounds: dict[str, object],
) -> ProteinHBond:
    a_value = eval_poly(coeff_a, angle)
    b_value = eval_poly(coeff_b, angle)
    c_value = eval_poly(coeff_c, angle)
    hbond_score = chemem_hbond_score(
        a_value,
        b_value,
        c_value,
        donor_acceptor_distance,
    )
    normalized_score = normalize_hbond_score(
        hbond_score,
        donor.atom_type,
        acceptor.atom_type,
        score_bounds,
    )
    return ProteinHBond(
        model_id=donor.atom.model_id,
        donor=donor.atom,
        hydrogen=hydrogen,
        acceptor=acceptor.atom,
        donor_type=donor.atom_type,
        acceptor_type=acceptor.atom_type,
        donor_acceptor_distance=donor_acceptor_distance,
        hydrogen_acceptor_distance=distance(hydrogen.xyz, acceptor.atom.xyz),
        dha_angle=angle,
        a_value=a_value,
        b_value=b_value,
        c_value=c_value,
        hbond_score=hbond_score,
        normalized_score=normalized_score,
    )


def _select_one_hbond_per_donor_hydrogen(
    hbonds: Sequence[ProteinHBond],
    mode: str,
) -> list[ProteinHBond]:
    mode = _validate_hbond_per_donor_hydrogen(mode)
    if mode is None:
        return list(hbonds)
    selected: dict[tuple[object, ...], ProteinHBond] = {}
    for hbond in hbonds:
        donor_hydrogen_key = _donor_hydrogen_key(hbond)
        current = selected.get(donor_hydrogen_key)
        if current is None or _hbond_selection_key(hbond, mode) < _hbond_selection_key(current, mode):
            selected[donor_hydrogen_key] = hbond
    selected_hbonds = list(selected.values())
    _sort_hbonds(selected_hbonds)
    return selected_hbonds


def _hbond_selection_key(hbond: ProteinHBond, mode: str) -> tuple[object, ...]:
    acceptor_key = _acceptor_sort_key(hbond)
    if mode == HBOND_PER_DONOR_HYDROGEN_BEST_DISTANCE:
        return (
            hbond.donor_acceptor_distance,
            -hbond.normalized_score,
            hbond.hydrogen_acceptor_distance,
            -hbond.dha_angle,
            acceptor_key,
        )
    if mode == HBOND_PER_DONOR_HYDROGEN_BEST_NORMALIZED_SCORE:
        return (
            -hbond.normalized_score,
            hbond.hbond_score,
            hbond.donor_acceptor_distance,
            hbond.hydrogen_acceptor_distance,
            -hbond.dha_angle,
            acceptor_key,
        )
    raise ValueError(f"unsupported hbond_per_donor_hydrogen mode: {mode}")


def _donor_hydrogen_key(hbond: ProteinHBond) -> tuple[object, ...]:
    return (
        hbond.model_id,
        hbond.donor.chain_id,
        hbond.donor.res_seq,
        hbond.donor.ins_code,
        hbond.donor.res_name,
        hbond.donor.name,
        hbond.hydrogen.name,
        str(hbond.hydrogen.atom_id),
    )


def _acceptor_sort_key(hbond: ProteinHBond) -> tuple[object, ...]:
    return (
        hbond.acceptor.model_id,
        hbond.acceptor.chain_id,
        hbond.acceptor.res_seq,
        hbond.acceptor.ins_code,
        hbond.acceptor.res_name,
        hbond.acceptor.name,
        str(hbond.acceptor.atom_id),
    )


def _sort_hbonds(hbonds: list[ProteinHBond]) -> None:
    hbonds.sort(
        key=lambda item: (
            item.model_id,
            item.donor.chain_id,
            item.donor.res_seq,
            item.donor.name,
            item.acceptor.chain_id,
            item.acceptor.res_seq,
            item.acceptor.name,
            str(item.acceptor.serial),
        )
    )


def _with_environment_context(
    structure: ProteinStructure,
    hbonds: list[ProteinHBond],
    *,
    workers: int = 1,
) -> list[ProteinHBond]:
    hbonds_with_context, _ = _with_environment_context_worker_count(
        structure,
        hbonds,
        workers=workers,
    )
    return hbonds_with_context


def _with_environment_context_worker_count(
    structure: ProteinStructure,
    hbonds: list[ProteinHBond],
    *,
    workers: int = 1,
) -> tuple[list[ProteinHBond], int]:
    if not hbonds:
        return hbonds, 0
    effective_workers = _effective_worker_count(workers, len(hbonds))
    if effective_workers == 1:
        return _context_hbond_chunk(structure, hbonds), effective_workers

    chunks = _chunk_sequence(hbonds, effective_workers)
    tasks = [(structure, chunk) for chunk in chunks]
    hbonds_with_context: list[ProteinHBond] = []
    try:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            for chunk_hbonds in executor.map(_context_hbond_chunk_from_args, tasks):
                hbonds_with_context.extend(chunk_hbonds)
    except OSError:
        effective_workers = 1
        hbonds_with_context = _context_hbond_chunk(structure, hbonds)
    return hbonds_with_context, effective_workers


def _context_hbond_chunk_from_args(args: tuple[ProteinStructure, Sequence[ProteinHBond]]) -> list[ProteinHBond]:
    structure, hbonds = args
    return _context_hbond_chunk(structure, hbonds)


def _context_hbond_chunk(
    structure: ProteinStructure,
    hbonds: Sequence[ProteinHBond],
) -> list[ProteinHBond]:
    calculator = FastEnvironmentContextCalculator(structure)
    return [
        replace(hbond, environment_context=calculator.context_for_hbond(hbond))
        for hbond in hbonds
    ]


def _validate_workers(workers: int) -> int:
    try:
        value = int(workers)
    except (TypeError, ValueError) as exc:
        raise ValueError("workers must be a positive integer") from exc
    if value < 1:
        raise ValueError("workers must be a positive integer")
    return value


def _validate_positive_float(value: float, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive number")
    return parsed


def _effective_worker_count(requested_workers: int, item_count: int) -> int:
    requested_workers = _validate_workers(requested_workers)
    if requested_workers == 1 or item_count < 2:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(requested_workers, cpu_count, item_count))


def _chunk_sequence(items: Sequence, chunk_count: int) -> list[tuple]:
    chunk_count = max(1, min(chunk_count, len(items)))
    chunk_size = math.ceil(len(items) / chunk_count)
    return [tuple(items[index : index + chunk_size]) for index in range(0, len(items), chunk_size)]


def _validate_context_mode(context_mode: str) -> None:
    if context_mode not in CONTEXT_MODES:
        modes = ", ".join(sorted(CONTEXT_MODES))
        raise ValueError(f"context_mode must be one of: {modes}")


def _validate_hbond_per_donor_hydrogen(value: str | None) -> str | None:
    if value in {None, "", "none"}:
        return None
    if value not in HBOND_PER_DONOR_HYDROGEN_MODES:
        modes = ", ".join(sorted(HBOND_PER_DONOR_HYDROGEN_MODES))
        raise ValueError(f"hbond_per_donor_hydrogen must be one of: {modes}")
    return value


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


def _has_relevant_explicit_donors(structure: ProteinStructure, selector: AtomSelector) -> bool:
    donors = find_protein_donors(structure)
    if not donors:
        return False
    if selector.all_atoms or any(selector.matches(donor.atom, donor.atom_type) for donor in donors):
        return True
    acceptors = find_protein_acceptors(structure)
    return any(selector.matches(acceptor.atom, acceptor.atom_type) for acceptor in acceptors)


def _build_spatial_grid(
    acceptors: Iterable[ProteinAcceptor],
    cell_size: float,
) -> dict[tuple[int, int, int], list[ProteinAcceptor]]:
    grid: dict[tuple[int, int, int], list[ProteinAcceptor]] = defaultdict(list)
    for acceptor in acceptors:
        grid[_grid_key(acceptor.atom.xyz, cell_size)].append(acceptor)
    return grid


def _nearby_acceptors(
    donor: Atom,
    grid: Mapping[tuple[int, int, int], list[ProteinAcceptor]],
    cell_size: float,
) -> Iterable[ProteinAcceptor]:
    cx, cy, cz = _grid_key(donor.xyz, cell_size)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield from grid.get((cx + dx, cy + dy, cz + dz), ())


def _same_atom(first: Atom, second: Atom) -> bool:
    return first.heavy_identity == second.heavy_identity


def _grid_key(coord: Sequence[float], cell_size: float) -> tuple[int, int, int]:
    return (
        math.floor(float(coord[0]) / cell_size),
        math.floor(float(coord[1]) / cell_size),
        math.floor(float(coord[2]) / cell_size),
    )
