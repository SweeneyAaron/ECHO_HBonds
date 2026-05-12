"""Write ChimeraX defattr attribute files for per-residue HBond visualisation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .environment_context import CONTEXT_FIELDS, CONTEXT_MODE_NONE
from .pdb_io import ResidueKey

if TYPE_CHECKING:
    from .backbone_amide import ProteinHBond, ScoreResult


SCORE_ATTRIBUTES = ("hbond_score", "normalized_score")
CXC_FILENAME = "hbond_chimerax.cxc"


def _snake_to_camel(name: str) -> str:
    head, *tail = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail)


def _residue_spec(key: ResidueKey) -> str:
    chain = key.chain_id or ""
    return f"/{chain}:{key.res_seq}{key.ins_code}"


def _best_bond_per_residue(
    hbonds: list["ProteinHBond"],
) -> dict[ResidueKey, "ProteinHBond"]:
    best: dict[ResidueKey, "ProteinHBond"] = {}
    for bond in hbonds:
        for atom in (bond.donor, bond.acceptor):
            key = atom.residue_key
            current = best.get(key)
            if current is None or bond.normalized_score > current.normalized_score:
                best[key] = bond
    return best


def _attribute_value(bond: "ProteinHBond", field: str) -> object:
    if field in SCORE_ATTRIBUTES:
        return getattr(bond, field)
    if bond.environment_context is None:
        return None
    return getattr(bond.environment_context, field)


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def _write_defattr(
    path: Path,
    attribute_name: str,
    entries: list[tuple[ResidueKey, object]],
) -> None:
    lines = [
        "# written by HBOND_CHEMEM --chimerax",
        "# best bond per residue, ranked by normalized_score",
        f"attribute: {attribute_name}",
        "recipient: residues",
        "match mode: any",
    ]
    for key, value in entries:
        lines.append(f"\t{_residue_spec(key)}\t{_format_value(value)}")
    path.write_text("\n".join(lines) + "\n")


def _sorted_entries(
    best_by_residue: dict[ResidueKey, "ProteinHBond"],
    field: str,
) -> list[tuple[ResidueKey, object]]:
    items: list[tuple[ResidueKey, object]] = []
    for key, bond in best_by_residue.items():
        value = _attribute_value(bond, field)
        if value is None:
            continue
        items.append((key, value))
    items.sort(
        key=lambda item: (
            item[0].model_id,
            item[0].chain_id,
            item[0].res_seq,
            item[0].ins_code,
        )
    )
    return items


def _write_cxc(
    path: Path,
    defattr_files: list[str],
    attribute_names: list[str],
) -> None:
    lines = [
        "# helper script for HBOND_CHEMEM --chimerax output",
        "# load this after opening the corresponding PDB in ChimeraX",
    ]
    for filename in defattr_files:
        lines.append(f"open {filename}")
    if attribute_names:
        primary = attribute_names[0]
        lines.append("")
        lines.append(f"color byattribute r:{primary} palette blue-white-red")
        if len(attribute_names) > 1:
            lines.append("# swap r:<name> with any of:")
            for name in attribute_names[1:]:
                lines.append(f"#   r:{name}")
    path.write_text("\n".join(lines) + "\n")


def write_chimerax(result: "ScoreResult", output_dir: str | Path) -> None:
    """Write per-residue defattr files plus a helper .cxc script.

    For each residue that participates (as donor or acceptor) in at least one
    scored HBond, the residue's value for every visualisable attribute comes
    from the single HBond touching that residue with the highest
    ``normalized_score``.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields = list(SCORE_ATTRIBUTES)
    if result.context_mode != CONTEXT_MODE_NONE:
        fields.extend(CONTEXT_FIELDS)

    best_by_residue = _best_bond_per_residue(list(result.hbonds))

    written_files: list[str] = []
    attribute_names: list[str] = []
    for field in fields:
        entries = _sorted_entries(best_by_residue, field) if best_by_residue else []
        if not entries:
            continue
        attribute_name = _snake_to_camel(field)
        filename = f"{field}.defattr"
        _write_defattr(out / filename, attribute_name, entries)
        written_files.append(filename)
        attribute_names.append(attribute_name)

    _write_cxc(out / CXC_FILENAME, written_files, attribute_names)
