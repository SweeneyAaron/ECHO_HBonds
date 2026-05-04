"""Protein donor/acceptor typing for ChemEM HBond scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .pdb_io import Atom, ProteinStructure, ResidueKey, WATER_RES_NAMES, normalize_atom_name


GENERIC_N_TYPE = 13
GENERIC_N_ONE_H_TYPE = 15
AROMATIC_N_ACCEPTOR_TYPE = 14
GENERIC_O_TYPE = 19
SULFUR_TYPE = 24
AMIDE_O_TYPE = 38
PEPTIDE_O_TYPE = 39
PEPTIDE_N_TYPE = 40
AMIDE_N_TYPE = 43


@dataclass(frozen=True)
class ProteinDonor:
    atom: Atom
    hydrogens: tuple[Atom, ...]
    atom_type: int


@dataclass(frozen=True)
class ProteinAcceptor:
    atom: Atom
    atom_type: int


@dataclass(frozen=True)
class SelectorTerm:
    raw: str
    type_id: int | None = None
    residue_name: str | None = None
    atom_name: str | None = None
    token: str | None = None

    def matches(self, atom: Atom, atom_type: int) -> bool:
        if self.type_id is not None:
            return atom_type == self.type_id
        if self.residue_name is not None and self.atom_name is not None:
            return (
                atom.res_name.upper() == self.residue_name
                and normalize_atom_name(atom.name) == self.atom_name
            )
        if self.token is None:
            return False
        return normalize_atom_name(atom.name) == self.token


@dataclass(frozen=True)
class AtomSelector:
    raw: str
    all_atoms: bool
    terms: tuple[SelectorTerm, ...]

    def matches(self, atom: Atom, atom_type: int) -> bool:
        return self.all_atoms or any(term.matches(atom, atom_type) for term in self.terms)


BACKBONE_H_NAMES = {
    "H",
    "HN",
    "H1",
    "H2",
    "H3",
    "1H",
    "2H",
    "3H",
    "HT1",
    "HT2",
    "HT3",
}

DONOR_DEFINITIONS: dict[tuple[str, str], tuple[int, frozenset[str]]] = {
    ("ASN", "ND2"): (AMIDE_N_TYPE, frozenset({"HD21", "HD22", "1HD2", "2HD2"})),
    ("GLN", "NE2"): (AMIDE_N_TYPE, frozenset({"HE21", "HE22", "1HE2", "2HE2"})),
    ("ARG", "NE"): (GENERIC_N_ONE_H_TYPE, frozenset({"HE"})),
    ("ARG", "NH1"): (GENERIC_N_TYPE, frozenset({"HH11", "HH12", "1HH1", "2HH1"})),
    ("ARG", "NH2"): (GENERIC_N_TYPE, frozenset({"HH21", "HH22", "1HH2", "2HH2"})),
    ("LYS", "NZ"): (GENERIC_N_TYPE, frozenset({"HZ1", "HZ2", "HZ3", "1HZ", "2HZ", "3HZ"})),
    ("TRP", "NE1"): (AMIDE_N_TYPE, frozenset({"HE1"})),
    ("SER", "OG"): (GENERIC_O_TYPE, frozenset({"HG"})),
    ("THR", "OG1"): (GENERIC_O_TYPE, frozenset({"HG1"})),
    ("TYR", "OH"): (GENERIC_O_TYPE, frozenset({"HH"})),
    ("CYS", "SG"): (SULFUR_TYPE, frozenset({"HG"})),
}

for _his_name in ("HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP"):
    DONOR_DEFINITIONS[(_his_name, "ND1")] = (AMIDE_N_TYPE, frozenset({"HD1"}))
    DONOR_DEFINITIONS[(_his_name, "NE2")] = (AMIDE_N_TYPE, frozenset({"HE2"}))

ACCEPTOR_DEFINITIONS: dict[tuple[str, str], int] = {
    ("ASN", "OD1"): AMIDE_O_TYPE,
    ("GLN", "OE1"): AMIDE_O_TYPE,
    ("SER", "OG"): GENERIC_O_TYPE,
    ("THR", "OG1"): GENERIC_O_TYPE,
    ("TYR", "OH"): GENERIC_O_TYPE,
    ("CYS", "SG"): SULFUR_TYPE,
    ("CYX", "SG"): SULFUR_TYPE,
    ("ASP", "OD1"): GENERIC_O_TYPE,
    ("ASP", "OD2"): GENERIC_O_TYPE,
    ("ASH", "OD1"): GENERIC_O_TYPE,
    ("ASH", "OD2"): GENERIC_O_TYPE,
    ("GLU", "OE1"): GENERIC_O_TYPE,
    ("GLU", "OE2"): GENERIC_O_TYPE,
    ("GLH", "OE1"): GENERIC_O_TYPE,
    ("GLH", "OE2"): GENERIC_O_TYPE,
    ("MET", "SD"): SULFUR_TYPE,
}

for _his_name in ("HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP"):
    ACCEPTOR_DEFINITIONS[(_his_name, "ND1")] = AROMATIC_N_ACCEPTOR_TYPE
    ACCEPTOR_DEFINITIONS[(_his_name, "NE2")] = AROMATIC_N_ACCEPTOR_TYPE


def parse_atom_selector(atom_types: str | Sequence[str] | None = None) -> AtomSelector:
    """Parse atom selector tokens such as ``N``, ``SER:OG``, ``40``, or ``ALL``."""

    raw = _selector_raw(atom_types)
    parts = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    if not parts:
        raise ValueError("atom_types must contain at least one selector")
    if any(part.upper() in {"ALL", "*"} for part in parts):
        return AtomSelector(raw=raw, all_atoms=True, terms=())

    terms: list[SelectorTerm] = []
    for part in parts:
        upper = part.upper()
        if upper.isdigit():
            terms.append(SelectorTerm(raw=part, type_id=int(upper)))
            continue
        if ":" in upper:
            residue, atom_name = upper.split(":", 1)
            residue = residue.strip()
            atom_name = normalize_atom_name(atom_name)
            if not residue or not atom_name:
                raise ValueError(f"invalid atom selector: {part}")
            terms.append(SelectorTerm(raw=part, residue_name=residue, atom_name=atom_name))
            continue
        token = normalize_atom_name(upper)
        if not token:
            raise ValueError(f"invalid atom selector: {part}")
        terms.append(SelectorTerm(raw=part, token=token))
    return AtomSelector(raw=raw, all_atoms=False, terms=tuple(terms))


def find_protein_donors(structure: ProteinStructure) -> list[ProteinDonor]:
    residues = _group_protein_atoms_by_residue(structure.atoms)
    donors: list[ProteinDonor] = []
    for atoms in residues.values():
        hydrogens = tuple(atom for atom in atoms if atom.element == "H")
        heavy_atoms = [atom for atom in atoms if atom.element != "H"]
        for atom in heavy_atoms:
            atom_name = normalize_atom_name(atom.name)
            if atom_name == "N" and atom.element == "N":
                donor_type = PEPTIDE_N_TYPE
                h_names = BACKBONE_H_NAMES
            else:
                definition = DONOR_DEFINITIONS.get((atom.res_name.upper(), atom_name))
                if definition is None:
                    continue
                donor_type, h_names = definition

            donor_hydrogens = _attached_hydrogens(atom, hydrogens, h_names)
            if donor_hydrogens:
                donors.append(ProteinDonor(atom=atom, hydrogens=donor_hydrogens, atom_type=donor_type))
    return donors


def find_protein_acceptors(structure: ProteinStructure) -> list[ProteinAcceptor]:
    residues = _group_protein_atoms_by_residue(structure.atoms)
    acceptors: list[ProteinAcceptor] = []
    for atoms in residues.values():
        hydrogens = tuple(atom for atom in atoms if atom.element == "H")
        for atom in atoms:
            if atom.element == "H":
                continue
            atom_name = normalize_atom_name(atom.name)
            if atom_name == "O" and atom.element == "O":
                acceptors.append(ProteinAcceptor(atom=atom, atom_type=PEPTIDE_O_TYPE))
                continue
            if atom_name == "OXT" and atom.element == "O":
                acceptors.append(ProteinAcceptor(atom=atom, atom_type=GENERIC_O_TYPE))
                continue

            atom_type = ACCEPTOR_DEFINITIONS.get((atom.res_name.upper(), atom_name))
            if atom_type is None:
                continue
            if atom_type == AROMATIC_N_ACCEPTOR_TYPE and _attached_hydrogens(atom, hydrogens, ()):
                continue
            acceptors.append(ProteinAcceptor(atom=atom, atom_type=atom_type))
    return acceptors


def _selector_raw(atom_types: str | Sequence[str] | None) -> str:
    if atom_types is None:
        return "N"
    if isinstance(atom_types, str):
        return atom_types.strip() or "N"
    return ",".join(str(item).strip() for item in atom_types if str(item).strip()) or "N"


def _attached_hydrogens(
    heavy_atom: Atom,
    hydrogens: Iterable[Atom],
    expected_names: Iterable[str],
) -> tuple[Atom, ...]:
    expected = {normalize_atom_name(name) for name in expected_names}
    max_distance2 = _max_hydrogen_distance(heavy_atom.element) ** 2
    attached: list[Atom] = []
    seen: set[int | str] = set()
    for hydrogen in hydrogens:
        h_name = normalize_atom_name(hydrogen.name)
        name_matches = h_name in expected
        distance_matches = _squared_distance(heavy_atom.xyz, hydrogen.xyz) <= max_distance2
        if not (name_matches or distance_matches):
            continue
        atom_id = hydrogen.atom_id
        if atom_id in seen:
            continue
        seen.add(atom_id)
        attached.append(hydrogen)
    return tuple(attached)


def _max_hydrogen_distance(element: str) -> float:
    if element == "O":
        return 1.25
    if element == "S":
        return 1.55
    return 1.35


def _squared_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return dx * dx + dy * dy + dz * dz


def _group_protein_atoms_by_residue(atoms: Iterable[Atom]) -> dict[ResidueKey, list[Atom]]:
    residues: dict[ResidueKey, list[Atom]] = {}
    for atom in atoms:
        if atom.record_name != "ATOM" or atom.res_name in WATER_RES_NAMES:
            continue
        residues.setdefault(atom.residue_key, []).append(atom)
    return residues
