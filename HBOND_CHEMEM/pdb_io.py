"""Small PDB parser for backbone amide HBond scoring."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable


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

WATER_RES_NAMES = {"HOH", "WAT", "H2O", "DOD"}


@dataclass(frozen=True)
class ResidueKey:
    model_id: int
    chain_id: str
    res_seq: int
    ins_code: str
    res_name: str


@dataclass(frozen=True)
class Atom:
    serial: int | None
    name: str
    element: str
    res_name: str
    chain_id: str
    res_seq: int
    ins_code: str
    model_id: int
    x: float
    y: float
    z: float
    record_name: str = "ATOM"
    altloc: str = ""
    generated: bool = False

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def residue_key(self) -> ResidueKey:
        return ResidueKey(
            self.model_id,
            self.chain_id,
            self.res_seq,
            self.ins_code,
            self.res_name,
        )

    @property
    def atom_id(self) -> int | str:
        if self.serial is not None and not self.generated:
            return self.serial
        return (
            f"generated:model:{self.model_id}:chain:{self.chain_id or '_'}:"
            f"{self.res_seq}:{self.ins_code or '_'}:{self.name}"
        )

    @property
    def heavy_identity(self) -> tuple[int, str, int, str, str, str]:
        return (
            self.model_id,
            self.chain_id,
            self.res_seq,
            self.ins_code,
            self.res_name,
            self.name,
        )


@dataclass(frozen=True)
class BackboneDonor:
    atom: Atom
    hydrogens: tuple[Atom, ...]


@dataclass(frozen=True)
class ProteinStructure:
    atoms: tuple[Atom, ...]

    def with_original_heavy_atom_ids(self, original: "ProteinStructure") -> "ProteinStructure":
        original_heavy = {
            atom.heavy_identity: atom.serial
            for atom in original.atoms
            if atom.element != "H" and atom.record_name == "ATOM"
        }
        original_h = {
            atom.heavy_identity
            for atom in original.atoms
            if atom.element == "H" and atom.record_name == "ATOM"
        }
        merged: list[Atom] = []
        for atom in self.atoms:
            if atom.element == "H":
                merged.append(replace(atom, generated=atom.heavy_identity not in original_h, serial=None))
            elif atom.heavy_identity in original_heavy:
                merged.append(replace(atom, serial=original_heavy[atom.heavy_identity]))
            else:
                merged.append(atom)
        return ProteinStructure(tuple(merged))


def parse_pdb(path: str | Path) -> ProteinStructure:
    atoms: list[Atom] = []
    model_id = 1
    saw_model = False
    for line in Path(path).read_text().splitlines():
        record = line[0:6].strip()
        if record == "MODEL":
            saw_model = True
            model_id = _safe_int(line[10:14].strip(), default=model_id)
            continue
        if record == "ENDMDL":
            if saw_model:
                model_id += 1
            continue
        if record not in {"ATOM", "HETATM"}:
            continue
        if len(line) < 54:
            continue

        altloc = line[16:17].strip()
        if altloc not in {"", "A"}:
            continue

        name = line[12:16].strip()
        res_name = line[17:20].strip()
        chain_id = line[21:22].strip()
        res_seq = _safe_int(line[22:26].strip(), default=0)
        ins_code = line[26:27].strip()
        element = line[76:78].strip() if len(line) >= 78 else ""
        element = normalize_element(element or infer_element(name))
        atoms.append(
            Atom(
                serial=_safe_int(line[6:11].strip(), default=None),
                name=name,
                element=element,
                res_name=res_name,
                chain_id=chain_id,
                res_seq=res_seq,
                ins_code=ins_code,
                model_id=model_id,
                x=float(line[30:38]),
                y=float(line[38:46]),
                z=float(line[46:54]),
                record_name=record,
                altloc=altloc,
            )
        )
    return ProteinStructure(tuple(atoms))


def find_backbone_donors(structure: ProteinStructure) -> list[BackboneDonor]:
    residue_atoms = _group_protein_atoms_by_residue(structure.atoms)
    donors: list[BackboneDonor] = []
    for atoms in residue_atoms.values():
        n_atoms = [atom for atom in atoms if atom.name == "N" and atom.element == "N"]
        if not n_atoms:
            continue
        hydrogens = tuple(
            atom
            for atom in atoms
            if atom.element == "H" and normalize_atom_name(atom.name) in BACKBONE_H_NAMES
        )
        if hydrogens:
            donors.append(BackboneDonor(n_atoms[0], hydrogens))
    return donors


def find_backbone_acceptors(structure: ProteinStructure) -> list[Atom]:
    return [
        atom
        for atom in structure.atoms
        if (
            atom.record_name == "ATOM"
            and atom.res_name not in WATER_RES_NAMES
            and atom.name == "O"
            and atom.element == "O"
        )
    ]


def normalize_atom_name(name: str) -> str:
    return name.strip().upper().replace(" ", "")


def normalize_element(element: str) -> str:
    value = element.strip().upper()
    if value in {"D", "T"}:
        return "H"
    if not value:
        return ""
    return value[0] + value[1:].lower()


def infer_element(atom_name: str) -> str:
    letters = "".join(ch for ch in atom_name.strip() if ch.isalpha())
    if not letters:
        return ""
    upper = letters.upper()
    if upper.startswith(("CL", "BR")):
        return upper[:2]
    return upper[0]


def _group_protein_atoms_by_residue(atoms: Iterable[Atom]) -> dict[ResidueKey, list[Atom]]:
    residues: dict[ResidueKey, list[Atom]] = {}
    for atom in atoms:
        if atom.record_name != "ATOM" or atom.res_name in WATER_RES_NAMES:
            continue
        residues.setdefault(atom.residue_key, []).append(atom)
    return residues


def _safe_int(value: str, *, default: int | None) -> int | None:
    try:
        return int(value)
    except ValueError:
        return default
