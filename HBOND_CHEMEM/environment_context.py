"""Fast local environment descriptors for scored protein HBonds."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

from .pdb_io import Atom, ProteinStructure, WATER_RES_NAMES


CONTEXT_MODE_FAST = "fast"
CONTEXT_MODE_NONE = "none"
CONTEXT_MODES = {CONTEXT_MODE_FAST, CONTEXT_MODE_NONE}

CONTEXT_FIELDS = [
    "env_h_sasa_fraction",
    "env_h_solvent_reach_fraction",
    "env_h_packing_count_6p5",
    "env_h_electrostatic",
    "env_h_hydrophobic",
    "env_mid_sasa_fraction",
    "env_mid_electrostatic",
    "env_mid_hydrophobic",
]


@dataclass(frozen=True)
class EnvironmentContext:
    """Local environment values attached to one scored HBond."""

    env_h_sasa_fraction: float
    env_h_solvent_reach_fraction: float
    env_h_packing_count_6p5: int
    env_h_electrostatic: float
    env_h_hydrophobic: float
    env_mid_sasa_fraction: float
    env_mid_electrostatic: float
    env_mid_hydrophobic: float

    def to_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EnvironmentContextConfig:
    """Constants for the dependency-light fast context approximation."""

    cell_size: float = 4.0
    probe_radius: float = 1.4
    hydrogen_surface_radius: float = 2.6
    midpoint_surface_radius: float = 1.4
    sasa_direction_count: int = 32
    solvent_reach_ray_count: int = 32
    solvent_reach_ray_length: float = 8.0
    packing_cutoff: float = 6.5
    electrostatic_cutoff: float = 10.0
    electrostatic_decay: float = 8.0
    electrostatic_min_r: float = 1.0
    hydrophobic_cutoff: float = 6.0
    hydrophobic_decay: float = 4.0

    @property
    def max_query_radius(self) -> float:
        return max(
            self.hydrogen_surface_radius + MAX_VDW_RADIUS + self.probe_radius,
            self.midpoint_surface_radius + MAX_VDW_RADIUS + self.probe_radius,
            self.solvent_reach_ray_length + MAX_VDW_RADIUS + self.probe_radius,
            self.packing_cutoff,
            self.electrostatic_cutoff,
            self.hydrophobic_cutoff,
        )


DEFAULT_CONTEXT_CONFIG = EnvironmentContextConfig()


VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
}
DEFAULT_VDW_RADIUS = 1.70
MAX_VDW_RADIUS = max(VDW_RADII.values())

ACIDIC_ATOM_CHARGES = {
    ("ASP", "OD1"): -0.50,
    ("ASP", "OD2"): -0.50,
    ("GLU", "OE1"): -0.50,
    ("GLU", "OE2"): -0.50,
}

BASIC_ATOM_CHARGES = {
    ("ARG", "NE"): 0.33,
    ("ARG", "NH1"): 0.33,
    ("ARG", "NH2"): 0.33,
    ("LYS", "NZ"): 1.00,
    ("HIP", "ND1"): 0.50,
    ("HIP", "NE2"): 0.50,
    ("HSP", "ND1"): 0.50,
    ("HSP", "NE2"): 0.50,
}

POLAR_ELEMENT_CHARGES = {
    "N": 0.12,
    "O": -0.12,
    "S": -0.05,
    "P": 0.25,
}

RESIDUE_HYDROPHOBICITY = {
    "ILE": 1.00,
    "VAL": 0.95,
    "LEU": 0.95,
    "PHE": 0.90,
    "TRP": 0.85,
    "MET": 0.75,
    "ALA": 0.55,
    "PRO": 0.50,
    "CYS": 0.45,
    "TYR": 0.25,
    "GLY": 0.10,
    "SER": -0.10,
    "THR": -0.10,
    "ASN": -0.25,
    "GLN": -0.25,
    "HIS": -0.20,
    "HID": -0.20,
    "HIE": -0.20,
    "HIP": -0.35,
    "HSD": -0.20,
    "HSE": -0.20,
    "HSP": -0.35,
    "ASP": -0.55,
    "GLU": -0.55,
    "LYS": -0.45,
    "ARG": -0.55,
}


class FastEnvironmentContextCalculator:
    """Point-sample ChemEM-grid-like context features around scored HBonds."""

    def __init__(
        self,
        structure: ProteinStructure,
        config: EnvironmentContextConfig = DEFAULT_CONTEXT_CONFIG,
    ) -> None:
        self.config = config
        self.index = AtomSpatialIndex(_context_atoms(structure.atoms), cell_size=config.cell_size)
        self.surface_directions = _unit_sphere_directions(config.sasa_direction_count)
        self.ray_directions = _unit_sphere_directions(config.solvent_reach_ray_count)

    def context_for_hbond(self, hbond) -> EnvironmentContext:
        excluded = {
            _atom_context_key(hbond.donor),
            _atom_context_key(hbond.hydrogen),
            _atom_context_key(hbond.acceptor),
        }
        h_point = hbond.hydrogen.xyz
        midpoint = _midpoint(hbond.hydrogen.xyz, hbond.acceptor.xyz)

        return EnvironmentContext(
            env_h_sasa_fraction=self.sasa_fraction(
                h_point,
                sample_radius=self.config.hydrogen_surface_radius,
                excluded_atom_keys=excluded,
            ),
            env_h_solvent_reach_fraction=self.solvent_reach_fraction(
                h_point,
                excluded_atom_keys=excluded,
            ),
            env_h_packing_count_6p5=self.packing_count(
                h_point,
                excluded_atom_keys=excluded,
            ),
            env_h_electrostatic=self.electrostatic_potential(
                h_point,
                excluded_atom_keys=excluded,
            ),
            env_h_hydrophobic=self.hydrophobic_field(
                h_point,
                excluded_atom_keys=excluded,
            ),
            env_mid_sasa_fraction=self.sasa_fraction(
                midpoint,
                sample_radius=self.config.midpoint_surface_radius,
                excluded_atom_keys=excluded,
            ),
            env_mid_electrostatic=self.electrostatic_potential(
                midpoint,
                excluded_atom_keys=excluded,
            ),
            env_mid_hydrophobic=self.hydrophobic_field(
                midpoint,
                excluded_atom_keys=excluded,
            ),
        )

    def sasa_fraction(
        self,
        point: Sequence[float],
        *,
        sample_radius: float,
        excluded_atom_keys: set[tuple[object, ...]],
    ) -> float:
        search_radius = sample_radius + MAX_VDW_RADIUS + self.config.probe_radius
        neighbors = tuple(self.index.nearby(point, search_radius, excluded_atom_keys))
        if not self.surface_directions:
            return 0.0

        open_count = 0
        for direction in self.surface_directions:
            sample = _add_scaled(point, direction, sample_radius)
            if not _sample_is_occluded(sample, neighbors, self.config.probe_radius):
                open_count += 1
        return open_count / len(self.surface_directions)

    def solvent_reach_fraction(
        self,
        point: Sequence[float],
        *,
        excluded_atom_keys: set[tuple[object, ...]],
    ) -> float:
        search_radius = (
            self.config.solvent_reach_ray_length + MAX_VDW_RADIUS + self.config.probe_radius
        )
        neighbors = tuple(self.index.nearby(point, search_radius, excluded_atom_keys))
        if not self.ray_directions:
            return 0.0

        open_count = 0
        for direction in self.ray_directions:
            if not _ray_is_occluded(
                point,
                direction,
                self.config.solvent_reach_ray_length,
                neighbors,
                self.config.probe_radius,
            ):
                open_count += 1
        return open_count / len(self.ray_directions)

    def packing_count(
        self,
        point: Sequence[float],
        *,
        excluded_atom_keys: set[tuple[object, ...]],
    ) -> int:
        return sum(
            1
            for atom in self.index.nearby(point, self.config.packing_cutoff, excluded_atom_keys)
            if atom.element != "H"
        )

    def electrostatic_potential(
        self,
        point: Sequence[float],
        *,
        excluded_atom_keys: set[tuple[object, ...]],
    ) -> float:
        total = 0.0
        for atom in self.index.nearby(point, self.config.electrostatic_cutoff, excluded_atom_keys):
            charge = _charge_proxy(atom)
            if charge == 0.0:
                continue
            r = max(math.sqrt(_squared_distance(point, atom.xyz)), self.config.electrostatic_min_r)
            total += charge * math.exp(-r / self.config.electrostatic_decay) / r
        return total

    def hydrophobic_field(
        self,
        point: Sequence[float],
        *,
        excluded_atom_keys: set[tuple[object, ...]],
    ) -> float:
        total = 0.0
        for atom in self.index.nearby(point, self.config.hydrophobic_cutoff, excluded_atom_keys):
            if atom.element == "H":
                continue
            value = _hydrophobic_proxy(atom)
            if value == 0.0:
                continue
            r = math.sqrt(_squared_distance(point, atom.xyz))
            total += value * math.exp(-r / self.config.hydrophobic_decay)
        return total


class AtomSpatialIndex:
    """Simple fixed-cell spatial index for repeated local atom queries."""

    def __init__(self, atoms: Iterable[Atom], *, cell_size: float) -> None:
        self.cell_size = float(cell_size)
        self.grid: dict[tuple[int, int, int], list[Atom]] = defaultdict(list)
        for atom in atoms:
            self.grid[self._key(atom.xyz)].append(atom)

    def nearby(
        self,
        point: Sequence[float],
        radius: float,
        excluded_atom_keys: set[tuple[object, ...]] | None = None,
    ) -> Iterable[Atom]:
        excluded_atom_keys = excluded_atom_keys or set()
        radius = float(radius)
        radius2 = radius * radius
        cell_radius = int(math.ceil(radius / self.cell_size))
        cx, cy, cz = self._key(point)
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                for dz in range(-cell_radius, cell_radius + 1):
                    for atom in self.grid.get((cx + dx, cy + dy, cz + dz), ()):
                        if _atom_context_key(atom) in excluded_atom_keys:
                            continue
                        if _squared_distance(point, atom.xyz) <= radius2:
                            yield atom

    def _key(self, coord: Sequence[float]) -> tuple[int, int, int]:
        return (
            math.floor(float(coord[0]) / self.cell_size),
            math.floor(float(coord[1]) / self.cell_size),
            math.floor(float(coord[2]) / self.cell_size),
        )


def fast_context_metadata(
    mode: str,
    config: EnvironmentContextConfig = DEFAULT_CONTEXT_CONFIG,
) -> dict[str, object]:
    if mode == CONTEXT_MODE_NONE:
        return {
            "mode": CONTEXT_MODE_NONE,
            "fields": [],
            "context_excludes_hbond_atoms": True,
        }
    return {
        "mode": CONTEXT_MODE_FAST,
        "fields": list(CONTEXT_FIELDS),
        "probe_radius": config.probe_radius,
        "hydrogen_surface_radius": config.hydrogen_surface_radius,
        "midpoint_surface_radius": config.midpoint_surface_radius,
        "sasa_direction_count": config.sasa_direction_count,
        "solvent_reach_ray_count": config.solvent_reach_ray_count,
        "solvent_reach_ray_length": config.solvent_reach_ray_length,
        "packing_cutoff": config.packing_cutoff,
        "electrostatic_cutoff": config.electrostatic_cutoff,
        "electrostatic_decay": config.electrostatic_decay,
        "electrostatic_min_r": config.electrostatic_min_r,
        "electrostatic_model": "formal-plus-polar atom charge proxy, arbitrary units",
        "hydrophobic_cutoff": config.hydrophobic_cutoff,
        "hydrophobic_decay": config.hydrophobic_decay,
        "hydrophobic_model": "residue-plus-element hydrophobicity proxy, arbitrary units",
        "context_excludes_hbond_atoms": True,
        "full_env_grids_runtime_dependency": False,
    }


def _context_atoms(atoms: Iterable[Atom]) -> Iterable[Atom]:
    for atom in atoms:
        if atom.record_name != "ATOM":
            continue
        if atom.res_name in WATER_RES_NAMES:
            continue
        yield atom


def _atom_context_key(atom: Atom) -> tuple[object, ...]:
    return atom.heavy_identity


def _sample_is_occluded(
    sample: Sequence[float],
    atoms: Iterable[Atom],
    probe_radius: float,
) -> bool:
    for atom in atoms:
        obstruction_radius = _vdw_radius(atom) + probe_radius
        if _squared_distance(sample, atom.xyz) <= obstruction_radius * obstruction_radius:
            return True
    return False


def _ray_is_occluded(
    origin: Sequence[float],
    direction: Sequence[float],
    ray_length: float,
    atoms: Iterable[Atom],
    probe_radius: float,
) -> bool:
    for atom in atoms:
        ox = float(atom.x) - float(origin[0])
        oy = float(atom.y) - float(origin[1])
        oz = float(atom.z) - float(origin[2])
        projection = ox * direction[0] + oy * direction[1] + oz * direction[2]
        if projection < 0.0 or projection > ray_length:
            continue
        perpendicular2 = ox * ox + oy * oy + oz * oz - projection * projection
        obstruction_radius = _vdw_radius(atom) + probe_radius
        if perpendicular2 <= obstruction_radius * obstruction_radius:
            return True
    return False


def _charge_proxy(atom: Atom) -> float:
    key = (atom.res_name.upper(), atom.name.strip().upper())
    if key in ACIDIC_ATOM_CHARGES:
        return ACIDIC_ATOM_CHARGES[key]
    if key in BASIC_ATOM_CHARGES:
        return BASIC_ATOM_CHARGES[key]
    return POLAR_ELEMENT_CHARGES.get(atom.element, 0.0)


def _hydrophobic_proxy(atom: Atom) -> float:
    residue_value = RESIDUE_HYDROPHOBICITY.get(atom.res_name.upper(), 0.0)
    if atom.element == "C":
        return 0.60 + 0.45 * residue_value
    if atom.element == "S":
        return 0.35 + 0.30 * residue_value
    if atom.element == "N":
        return -0.25 + 0.10 * residue_value
    if atom.element == "O":
        return -0.35 + 0.10 * residue_value
    if atom.element == "P":
        return -0.40
    return 0.0


def _vdw_radius(atom: Atom) -> float:
    return VDW_RADII.get(atom.element, DEFAULT_VDW_RADIUS)


def _unit_sphere_directions(count: int) -> tuple[tuple[float, float, float], ...]:
    if count <= 0:
        return ()
    directions: list[tuple[float, float, float]] = []
    increment = math.pi * (3.0 - math.sqrt(5.0))
    offset = 2.0 / count
    for index in range(count):
        y = ((index * offset) - 1.0) + (offset / 2.0)
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        phi = index * increment
        directions.append((math.cos(phi) * radius, y, math.sin(phi) * radius))
    return tuple(directions)


def _midpoint(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        (float(a[0]) + float(b[0])) * 0.5,
        (float(a[1]) + float(b[1])) * 0.5,
        (float(a[2]) + float(b[2])) * 0.5,
    )


def _add_scaled(
    point: Sequence[float],
    direction: Sequence[float],
    scale: float,
) -> tuple[float, float, float]:
    return (
        float(point[0]) + direction[0] * scale,
        float(point[1]) + direction[1] * scale,
        float(point[2]) + direction[2] * scale,
    )


def _squared_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return dx * dx + dy * dy + dz * dz
