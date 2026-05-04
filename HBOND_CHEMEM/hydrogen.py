"""Hydrogen preparation helpers."""

from __future__ import annotations

from pathlib import Path


class MissingHydrogenDependencyError(RuntimeError):
    """Raised when PDBFixer/OpenMM are needed but unavailable."""


def add_hydrogens_with_pdbfixer(
    input_path: str | Path,
    output_path: str | Path,
    *,
    ph: float = 7.0,
) -> None:
    """Write a copy of ``input_path`` with missing hydrogens added by PDBFixer."""

    try:
        from pdbfixer import PDBFixer
    except ImportError as exc:
        raise MissingHydrogenDependencyError(
            "PDBFixer is required to add missing hydrogens. Install the documented "
            "Python 3.12 environment, or run with --hydrogen-mode explicit on a "
            "PDB that already contains hydrogens."
        ) from exc

    try:
        from openmm.app import PDBFile
    except ImportError:
        try:
            from simtk.openmm.app import PDBFile
        except ImportError as exc:
            raise MissingHydrogenDependencyError(
                "OpenMM is required to write the PDBFixer-hydrogenated structure."
            ) from exc

    fixer = PDBFixer(filename=str(input_path))
    fixer.addMissingHydrogens(ph)

    with Path(output_path).open("w") as handle:
        try:
            PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
        except TypeError:
            PDBFile.writeFile(fixer.topology, fixer.positions, handle)
