"""Hydrogen preparation helpers."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import urlopen


HYDROGEN_MINIMIZE_AUTO = "auto"
HYDROGEN_MINIMIZE_NONE = "none"
HYDROGEN_MINIMIZE_RESTRAINED = "restrained"
HYDROGEN_MINIMIZE_MODES = {
    HYDROGEN_MINIMIZE_AUTO,
    HYDROGEN_MINIMIZE_NONE,
    HYDROGEN_MINIMIZE_RESTRAINED,
}
HYDROGEN_FORCEFIELD_AUTO = "auto"
HYDROGEN_FORCEFIELD_AMBER = "amber"
HYDROGEN_FORCEFIELD_CHARMM = "charmm"
HYDROGEN_FORCEFIELD_MODES = {
    HYDROGEN_FORCEFIELD_AUTO,
    HYDROGEN_FORCEFIELD_AMBER,
    HYDROGEN_FORCEFIELD_CHARMM,
}
CCD_ONLINE_AUTO = "auto"
CCD_ONLINE_ALWAYS = "always"
CCD_ONLINE_NEVER = "never"
CCD_ONLINE_MODES = {CCD_ONLINE_AUTO, CCD_ONLINE_ALWAYS, CCD_ONLINE_NEVER}
HYDROGEN_MINIMIZATION_MAX_ITERATIONS = 50
HEAVY_ATOM_RESTRAINT_KJ_MOL_NM2 = 1000.0
AMBER_FORCEFIELD_FILES = ("amber14-all.xml", "amber14/tip3pfb.xml")
CHARMM_FORCEFIELD_FILES = ("charmm36_2024.xml", "charmm36_2024/water.xml")
CCD_URL_TEMPLATE = "https://files.rcsb.org/ligands/view/{code}.cif"
CCD_TIMEOUT_SECONDS = 10
STANDARD_PROTEIN_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "HID",
    "HIE",
    "HIP",
    "HSD",
    "HSE",
    "HSP",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}
WATER_RESIDUES = {"HOH", "WAT", "H2O", "DOD", "TIP3"}


class MissingHydrogenDependencyError(RuntimeError):
    """Raised when PDBFixer/OpenMM are needed but unavailable."""


class HydrogenMinimizationError(RuntimeError):
    """Raised when OpenMM cannot set up the requested minimization."""

    def __init__(
        self,
        message: str,
        *,
        attempted_forcefields: list[str] | None = None,
        forcefield_errors: list[dict[str, object]] | None = None,
    ) -> None:
        super().__init__(message)
        self.attempted_forcefields = attempted_forcefields or []
        self.forcefield_errors = forcefield_errors or []


def add_hydrogens_with_pdbfixer(
    input_path: str | Path,
    output_path: str | Path,
    *,
    ph: float = 7.0,
    minimize: str = HYDROGEN_MINIMIZE_AUTO,
    hydrogen_forcefield: str = HYDROGEN_FORCEFIELD_AUTO,
    ccd_cache: str | Path | None = None,
    ccd_online: str = CCD_ONLINE_AUTO,
) -> dict[str, object]:
    """Write a copy of ``input_path`` with missing hydrogens added by PDBFixer."""

    minimize = validate_hydrogen_minimize(minimize)
    hydrogen_forcefield = validate_hydrogen_forcefield(hydrogen_forcefield)
    ccd_online = validate_ccd_online(ccd_online)
    pdbfixer_module, PDBFixer = _load_pdbfixer()
    openmm_handles = _load_openmm_handles(include_minimizer=minimize != HYDROGEN_MINIMIZE_NONE)

    fixer = PDBFixer(filename=str(input_path))
    fixer.addMissingHydrogens(ph)
    metadata = hydrogen_minimization_metadata(
        minimize,
        ran=False,
        requested_forcefield=hydrogen_forcefield,
        ccd_cache=_metadata_path(_default_ccd_cache_dir(ccd_cache)),
        ccd_online=ccd_online,
        openmm_version=_module_version(openmm_handles.openmm),
        pdbfixer_version=_module_version(pdbfixer_module),
    )

    if minimize != HYDROGEN_MINIMIZE_NONE:
        try:
            preparation = _minimize_hydrogens_with_restrained_heavy_atoms(
                fixer,
                openmm_handles,
                ph=ph,
                hydrogen_forcefield=hydrogen_forcefield,
                ccd_cache=ccd_cache,
                ccd_online=ccd_online,
            )
        except HydrogenMinimizationError as exc:
            if minimize != HYDROGEN_MINIMIZE_AUTO:
                raise
            metadata = hydrogen_minimization_metadata(
                minimize,
                ran=False,
                requested_forcefield=hydrogen_forcefield,
                attempted_forcefields=exc.attempted_forcefields,
                forcefield_errors=exc.forcefield_errors,
                skipped_reason=str(exc),
                ccd_cache=_metadata_path(_default_ccd_cache_dir(ccd_cache)),
                ccd_online=ccd_online,
                openmm_version=_module_version(openmm_handles.openmm),
                pdbfixer_version=_module_version(pdbfixer_module),
            )
        else:
            metadata = hydrogen_minimization_metadata(
                minimize,
                ran=True,
                platform=preparation.get("platform"),
                selected_forcefield=preparation.get("selected_forcefield"),
                requested_forcefield=hydrogen_forcefield,
                attempted_forcefields=preparation.get("attempted_forcefields"),
                forcefield_files=preparation.get("forcefield_files"),
                forcefield_errors=preparation.get("forcefield_errors"),
                charmm_hydrogens_added=preparation.get("charmm_hydrogens_added"),
                unmatched_residues=preparation.get("unmatched_residues"),
                ccd_lookups=preparation.get("ccd_lookups"),
                ccd_cache=_metadata_path(_default_ccd_cache_dir(ccd_cache)),
                ccd_online=ccd_online,
                openmm_version=_module_version(openmm_handles.openmm),
                pdbfixer_version=_module_version(pdbfixer_module),
            )

    with Path(output_path).open("w") as handle:
        try:
            openmm_handles.PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
        except TypeError:
            openmm_handles.PDBFile.writeFile(fixer.topology, fixer.positions, handle)
    return metadata


def validate_hydrogen_minimize(minimize: str) -> str:
    if minimize not in HYDROGEN_MINIMIZE_MODES:
        modes = ", ".join(sorted(HYDROGEN_MINIMIZE_MODES))
        raise ValueError(f"hydrogen_minimize must be one of: {modes}")
    return minimize


def validate_hydrogen_forcefield(forcefield: str) -> str:
    if forcefield not in HYDROGEN_FORCEFIELD_MODES:
        modes = ", ".join(sorted(HYDROGEN_FORCEFIELD_MODES))
        raise ValueError(f"hydrogen_forcefield must be one of: {modes}")
    return forcefield


def validate_ccd_online(ccd_online: str) -> str:
    if ccd_online not in CCD_ONLINE_MODES:
        modes = ", ".join(sorted(CCD_ONLINE_MODES))
        raise ValueError(f"ccd_online must be one of: {modes}")
    return ccd_online


def hydrogen_minimization_metadata(
    minimize: str,
    *,
    ran: bool = False,
    platform: str | None = None,
    selected_forcefield: str | None = None,
    requested_forcefield: str | None = None,
    attempted_forcefields: list[str] | None = None,
    forcefield_files: tuple[str, ...] | list[str] | None = None,
    forcefield_errors: list[dict[str, object]] | None = None,
    skipped_reason: str | None = None,
    charmm_hydrogens_added: list[dict[str, object]] | None = None,
    unmatched_residues: list[dict[str, object]] | None = None,
    ccd_lookups: list[dict[str, object]] | None = None,
    ccd_cache: str | None = None,
    ccd_online: str | None = None,
    openmm_version: str | None = None,
    pdbfixer_version: str | None = None,
) -> dict[str, object]:
    return {
        "mode": validate_hydrogen_minimize(minimize),
        "ran": ran,
        "selected_forcefield": selected_forcefield,
        "requested_forcefield": (
            validate_hydrogen_forcefield(requested_forcefield)
            if requested_forcefield is not None
            else None
        ),
        "attempted_forcefields": attempted_forcefields or [],
        "forcefield_files": list(forcefield_files or []),
        "forcefield_errors": forcefield_errors or [],
        "skipped_reason": skipped_reason,
        "max_iterations": HYDROGEN_MINIMIZATION_MAX_ITERATIONS,
        "heavy_atom_restraint_kj_mol_nm2": HEAVY_ATOM_RESTRAINT_KJ_MOL_NM2,
        "platform": platform,
        "charmm_hydrogens_added": charmm_hydrogens_added or [],
        "unmatched_residues": unmatched_residues or [],
        "ccd_lookups": ccd_lookups or [],
        "ccd_cache": ccd_cache,
        "ccd_online": validate_ccd_online(ccd_online) if ccd_online is not None else None,
        "openmm_version": openmm_version,
        "pdbfixer_version": pdbfixer_version,
    }


def _load_pdbfixer():
    try:
        import pdbfixer as pdbfixer_module
        from pdbfixer import PDBFixer
    except ImportError as exc:
        raise MissingHydrogenDependencyError(
            "PDBFixer is required to add missing hydrogens. Install the documented "
            "Python 3.12 environment, or run with --hydrogen-mode explicit on a "
            "PDB that already contains hydrogens."
        ) from exc
    return pdbfixer_module, PDBFixer


def _load_openmm_handles(*, include_minimizer: bool):
    try:
        import openmm
        from openmm import CustomExternalForce, LocalEnergyMinimizer, Platform, VerletIntegrator, unit
        from openmm.app import ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation
    except ImportError:
        try:
            import simtk.openmm as openmm
            from simtk import unit
            from simtk.openmm import (
                CustomExternalForce,
                LocalEnergyMinimizer,
                Platform,
                VerletIntegrator,
            )
            from simtk.openmm.app import ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation
        except ImportError as exc:
            raise MissingHydrogenDependencyError(
                "OpenMM is required to write and minimize the PDBFixer-hydrogenated structure."
            ) from exc

    if include_minimizer:
        missing = [
            name
            for name, value in (
                ("CustomExternalForce", CustomExternalForce),
                ("ForceField", ForceField),
                ("LocalEnergyMinimizer", LocalEnergyMinimizer),
                ("Modeller", Modeller),
                ("Simulation", Simulation),
            )
            if value is None
        ]
        if missing:
            raise MissingHydrogenDependencyError(
                "OpenMM minimization requires: " + ", ".join(missing)
            )
    return SimpleNamespace(
        openmm=openmm,
        unit=unit,
        CustomExternalForce=CustomExternalForce,
        ForceField=ForceField,
        HBonds=HBonds,
        LocalEnergyMinimizer=LocalEnergyMinimizer,
        Modeller=Modeller,
        NoCutoff=NoCutoff,
        PDBFile=PDBFile,
        Platform=Platform,
        Simulation=Simulation,
        VerletIntegrator=VerletIntegrator,
    )


def _minimize_hydrogens_with_restrained_heavy_atoms(
    fixer,
    handles,
    *,
    ph: float,
    hydrogen_forcefield: str,
    ccd_cache: str | Path | None,
    ccd_online: str,
) -> dict[str, object]:
    attempted: list[str] = []
    errors: list[dict[str, object]] = []
    candidates = _forcefield_candidates(hydrogen_forcefield)
    for candidate in candidates:
        attempted.append(candidate)
        try:
            preparation = _prepare_forcefield_topology(
                fixer,
                handles,
                candidate,
                ph=ph,
                ccd_cache=ccd_cache,
                ccd_online=ccd_online,
            )
            platform_name, minimized_positions = _run_restrained_minimization(
                preparation["topology"],
                preparation["positions"],
                preparation["forcefield"],
                handles,
            )
        except Exception as exc:
            errors.append({"forcefield": candidate, "error": str(exc)})
            continue

        fixer.topology = preparation["topology"]
        fixer.positions = minimized_positions
        return {
            "selected_forcefield": candidate,
            "attempted_forcefields": attempted,
            "forcefield_errors": errors,
            "forcefield_files": preparation["forcefield_files"],
            "platform": platform_name,
            "charmm_hydrogens_added": preparation.get("charmm_hydrogens_added", []),
            "unmatched_residues": preparation.get("unmatched_residues", []),
            "ccd_lookups": preparation.get("ccd_lookups", []),
        }

    detail = "; ".join(f"{item['forcefield']}: {item['error']}" for item in errors)
    if not detail:
        detail = "no force fields were attempted"
    raise HydrogenMinimizationError(
        "OpenMM could not run restrained hydrogen minimization with "
        f"{', '.join(candidates)}: {detail}. This usually means the structure "
        "contains atoms or residues that are not covered by the selected "
        "force-field templates. Try --hydrogen-minimize none, or use "
        "--hydrogen-minimize auto to allow a no-minimization fallback.",
        attempted_forcefields=attempted,
        forcefield_errors=errors,
    )


def _forcefield_candidates(hydrogen_forcefield: str) -> list[str]:
    hydrogen_forcefield = validate_hydrogen_forcefield(hydrogen_forcefield)
    if hydrogen_forcefield == HYDROGEN_FORCEFIELD_AUTO:
        return [HYDROGEN_FORCEFIELD_AMBER, HYDROGEN_FORCEFIELD_CHARMM]
    return [hydrogen_forcefield]


def _prepare_forcefield_topology(
    fixer,
    handles,
    forcefield_name: str,
    *,
    ph: float,
    ccd_cache: str | Path | None,
    ccd_online: str,
) -> dict[str, object]:
    if forcefield_name == HYDROGEN_FORCEFIELD_AMBER:
        return {
            "topology": fixer.topology,
            "positions": fixer.positions,
            "forcefield": handles.ForceField(*AMBER_FORCEFIELD_FILES),
            "forcefield_files": AMBER_FORCEFIELD_FILES,
            "charmm_hydrogens_added": [],
            "unmatched_residues": [],
            "ccd_lookups": [],
        }
    if forcefield_name == HYDROGEN_FORCEFIELD_CHARMM:
        forcefield = handles.ForceField(*CHARMM_FORCEFIELD_FILES)
        charmm_preparation = _prepare_charmm_topology(
            fixer,
            handles,
            forcefield,
            ph=ph,
            ccd_cache=ccd_cache,
            ccd_online=ccd_online,
        )
        return {
            "topology": charmm_preparation["topology"],
            "positions": charmm_preparation["positions"],
            "forcefield": forcefield,
            "forcefield_files": CHARMM_FORCEFIELD_FILES,
            "charmm_hydrogens_added": charmm_preparation["charmm_hydrogens_added"],
            "unmatched_residues": charmm_preparation["unmatched_residues"],
            "ccd_lookups": charmm_preparation["ccd_lookups"],
        }
    raise ValueError(f"unsupported hydrogen_forcefield: {forcefield_name}")


def _prepare_charmm_topology(
    fixer,
    handles,
    forcefield,
    *,
    ph: float,
    ccd_cache: str | Path | None,
    ccd_online: str,
) -> dict[str, object]:
    variants, hydrogens_added = _charmm_template_hydrogen_variants(fixer.topology, forcefield)
    diagnostics = _charmm_residue_diagnostics(
        fixer.topology,
        forcefield,
        ccd_cache=ccd_cache,
        ccd_online=ccd_online,
    )

    if hydrogens_added:
        modeller = handles.Modeller(fixer.topology, fixer.positions)
        platform = _deterministic_platform(handles.Platform)
        state = random.getstate()
        random.seed(0)
        try:
            modeller.addHydrogens(forcefield, pH=ph, variants=variants, platform=platform)
        finally:
            random.setstate(state)
        topology = modeller.topology
        positions = modeller.positions
    else:
        topology = fixer.topology
        positions = fixer.positions

    return {
        "topology": topology,
        "positions": positions,
        "charmm_hydrogens_added": hydrogens_added,
        "unmatched_residues": diagnostics["unmatched_residues"],
        "ccd_lookups": diagnostics["ccd_lookups"],
    }


def _charmm_template_hydrogen_variants(topology, forcefield) -> tuple[list[object], list[dict[str, object]]]:
    variants: list[object] = []
    hydrogens_added: list[dict[str, object]] = []
    templates = getattr(forcefield, "_templates", {})
    for residue in topology.residues():
        template = templates.get(residue.name)
        hydrogens: list[tuple[str, str]] = []
        if template is not None and _should_add_charmm_template_hydrogens(residue):
            hydrogens = _missing_template_hydrogens(residue, template)
            if hydrogens:
                hydrogens_added.append(
                    {
                        **_residue_metadata(residue),
                        "hydrogens": [
                            {"name": name, "parent": parent}
                            for name, parent in hydrogens
                        ],
                    }
                )
        variants.append(hydrogens if hydrogens else None)
    return variants, hydrogens_added


def _should_add_charmm_template_hydrogens(residue) -> bool:
    name = residue.name.upper()
    return name not in STANDARD_PROTEIN_RESIDUES and name not in WATER_RESIDUES


def _missing_template_hydrogens(residue, template) -> list[tuple[str, str]]:
    existing_names = {atom.name for atom in residue.atoms()}
    residue_heavy_names = {
        atom.name
        for atom in residue.atoms()
        if _element_symbol(atom).upper() != "H"
    }
    template_heavy_names = {
        atom.name
        for atom in template.atoms
        if _element_symbol(atom).upper() != "H"
    }
    if residue_heavy_names != template_heavy_names:
        return []

    atom_indices = getattr(template, "atomIndices", {})
    missing: list[tuple[str, str]] = []
    for atom in template.atoms:
        if _element_symbol(atom).upper() != "H" or atom.name in existing_names:
            continue
        parent_name = _template_hydrogen_parent_name(atom, template, atom_indices)
        if parent_name is not None and parent_name in existing_names:
            missing.append((atom.name, parent_name))
    return missing


def _template_hydrogen_parent_name(atom, template, atom_indices: dict[str, int]) -> str | None:
    atom_index = atom_indices.get(atom.name)
    if atom_index is None:
        return None
    for first, second in template.bonds:
        if first == atom_index:
            candidate = template.atoms[second]
        elif second == atom_index:
            candidate = template.atoms[first]
        else:
            continue
        if _element_symbol(candidate).upper() != "H":
            return candidate.name
    return None


def _charmm_residue_diagnostics(
    topology,
    forcefield,
    *,
    ccd_cache: str | Path | None,
    ccd_online: str,
) -> dict[str, list[dict[str, object]]]:
    templates = getattr(forcefield, "_templates", {})
    unmatched: list[dict[str, object]] = []
    lookups: list[dict[str, object]] = []
    lookup_by_code: dict[str, dict[str, object]] = {}
    cache_dir = _default_ccd_cache_dir(ccd_cache)
    for residue in topology.residues():
        if not _should_add_charmm_template_hydrogens(residue):
            continue
        template = templates.get(residue.name)
        if template is None:
            reason = "no_charmm_template"
        elif not _template_heavy_atoms_match_residue(residue, template):
            reason = "heavy_atom_mismatch"
        else:
            continue
        residue_info = {**_residue_metadata(residue), "reason": reason}
        unmatched.append(residue_info)
        code = residue.name.upper()
        if code not in lookup_by_code:
            lookup_by_code[code] = _lookup_ccd_component(
                residue.name,
                cache_dir=cache_dir,
                ccd_online=ccd_online,
            )
        lookup = lookup_by_code[code]
        lookups.append({**_residue_metadata(residue), **lookup})
    return {"unmatched_residues": unmatched, "ccd_lookups": lookups}


def _template_heavy_atoms_match_residue(residue, template) -> bool:
    residue_heavy_names = {
        atom.name
        for atom in residue.atoms()
        if _element_symbol(atom).upper() != "H"
    }
    template_heavy_names = {
        atom.name
        for atom in template.atoms
        if _element_symbol(atom).upper() != "H"
    }
    return residue_heavy_names == template_heavy_names


def _lookup_ccd_component(
    code: str,
    *,
    cache_dir: Path,
    ccd_online: str,
) -> dict[str, object]:
    code = code.strip().upper()
    ccd_online = validate_ccd_online(ccd_online)
    cache_path = cache_dir / f"{code}.cif"
    url = CCD_URL_TEMPLATE.format(code=code)
    if cache_path.exists() and ccd_online != CCD_ONLINE_ALWAYS:
        return {
            "code": code,
            "available": True,
            "source": "cache",
            "path": str(cache_path),
            "url": url,
            "error": None,
        }
    if ccd_online != CCD_ONLINE_NEVER:
        try:
            with urlopen(url, timeout=CCD_TIMEOUT_SECONDS) as response:
                data = response.read()
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(data)
            return {
                "code": code,
                "available": True,
                "source": "rcsb",
                "path": str(cache_path),
                "url": url,
                "error": None,
            }
        except (OSError, URLError) as exc:
            if cache_path.exists():
                return {
                    "code": code,
                    "available": True,
                    "source": "cache",
                    "path": str(cache_path),
                    "url": url,
                    "error": str(exc),
                }
            return {
                "code": code,
                "available": False,
                "source": None,
                "path": str(cache_path),
                "url": url,
                "error": str(exc),
            }
    return {
        "code": code,
        "available": False,
        "source": None,
        "path": str(cache_path),
        "url": url,
        "error": "ccd_online is never and no cached CCD file is available",
    }


def _run_restrained_minimization(topology, positions, forcefield, handles) -> tuple[str | None, object]:
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=handles.NoCutoff,
        constraints=handles.HBonds,
    )
    system.addForce(_heavy_atom_restraint_force(topology, positions, handles))
    integrator = handles.VerletIntegrator(0.001 * handles.unit.picoseconds)
    platform = _deterministic_platform(handles.Platform)
    if platform is None:
        simulation = handles.Simulation(topology, system, integrator)
        platform_name = None
    else:
        simulation = handles.Simulation(topology, system, integrator, platform)
        platform_name = platform.getName()
    simulation.context.setPositions(positions)
    handles.LocalEnergyMinimizer.minimize(
        simulation.context,
        tolerance=10.0,
        maxIterations=HYDROGEN_MINIMIZATION_MAX_ITERATIONS,
    )
    minimized_positions = simulation.context.getState(getPositions=True).getPositions()
    return platform_name, minimized_positions


def _heavy_atom_restraint_force(topology, positions, handles):
    force = handles.CustomExternalForce(
        "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    )
    force.addGlobalParameter("k", HEAVY_ATOM_RESTRAINT_KJ_MOL_NM2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    for index, atom in enumerate(topology.atoms()):
        element_symbol = _element_symbol(atom)
        if element_symbol.upper() == "H":
            continue
        x0, y0, z0 = _position_components_nanometers(positions[index], handles.unit)
        force.addParticle(index, [x0, y0, z0])
    return force


def _residue_metadata(residue) -> dict[str, object]:
    chain = getattr(residue, "chain", None)
    return {
        "index": getattr(residue, "index", None),
        "name": getattr(residue, "name", None),
        "chain_id": getattr(chain, "id", ""),
        "residue_id": getattr(residue, "id", ""),
        "insertion_code": getattr(residue, "insertionCode", ""),
    }


def _element_symbol(atom) -> str:
    element = getattr(atom, "element", None)
    return str(getattr(element, "symbol", "") or "")


def _default_ccd_cache_dir(ccd_cache: str | Path | None) -> Path:
    if ccd_cache is not None:
        return Path(ccd_cache)
    return Path.home() / ".cache" / "hbond-chemem" / "ccd"


def _metadata_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path)


def _position_components_nanometers(position, unit_module) -> list[float]:
    try:
        value = position.value_in_unit(unit_module.nanometer)
    except AttributeError:
        value = position
    return [float(value[0]), float(value[1]), float(value[2])]


def _deterministic_platform(platform_class):
    for name in ("CPU", "Reference"):
        try:
            return platform_class.getPlatformByName(name)
        except Exception:
            continue
    return None


def _module_version(module) -> str | None:
    value = getattr(module, "__version__", None)
    if value is None:
        return None
    return str(value)
