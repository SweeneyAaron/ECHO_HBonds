"""Command line interface for protein HBond scoring."""

from __future__ import annotations

import argparse
import math
import sys

from .backbone_amide import (
    HBOND_PER_DONOR_HYDROGEN_MODES,
    score_pdb,
    write_csv,
    write_json,
)
from .environment_context import CONTEXT_MODE_FAST, CONTEXT_MODE_NONE
from .hydrogen import (
    CCD_ONLINE_AUTO,
    CCD_ONLINE_MODES,
    HYDROGEN_FORCEFIELD_AUTO,
    HYDROGEN_FORCEFIELD_MODES,
    HYDROGEN_MINIMIZE_AUTO,
    HYDROGEN_MINIMIZE_MODES,
    add_hydrogens_with_pdbfixer,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "score":
            result = score_pdb(
                args.input_pdb,
                hydrogen_mode=args.hydrogen_mode,
                ph=args.ph,
                atom_types=args.atom_types,
                context_mode=args.context_mode,
                workers=args.workers,
                distance_cutoff=args.hbond_distance_cutoff,
                hbond_per_donor_hydrogen=args.hbond_per_donor_hydrogen,
                hydrogen_minimize=args.hydrogen_minimize,
                hydrogen_forcefield=args.hydrogen_forcefield,
                ccd_cache=args.ccd_cache,
                ccd_online=args.ccd_online,
            )
            write_json(result, args.json)
            write_csv(result, args.csv)
            print(
                f"Scored {result.counts['hbonds']} HBonds "
                f"from {result.counts['donors']} donors and "
                f"{result.counts['acceptors']} acceptors "
                f"(selector: {result.atom_types})."
            )
            print(
                "Timing seconds: "
                f"parse={result.timing_seconds['parse']:.6f}, "
                f"hydrogen={result.timing_seconds['hydrogen']:.6f}, "
                f"score={result.timing_seconds['score']:.6f}, "
                f"context={result.timing_seconds['context']:.6f}, "
                f"total={result.timing_seconds['total']:.6f}"
            )
            return 0
        if args.command == "prepare":
            add_hydrogens_with_pdbfixer(
                args.input_pdb,
                args.output,
                ph=args.ph,
                minimize=args.hydrogen_minimize,
                hydrogen_forcefield=args.hydrogen_forcefield,
                ccd_cache=args.ccd_cache,
                ccd_online=args.ccd_online,
            )
            print(f"Wrote hydrogenated PDB to {args.output}")
            return 0
        if args.command == "batch":
            raise NotImplementedError(
                "Batch scoring is a stub for now because the batch input format is not finalized."
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m HBOND_CHEMEM",
        description="Score protein HBonds with ChemEM HBond scoring functions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="score one protein PDB")
    score_parser.add_argument("input_pdb", help="input PDB file")
    score_parser.add_argument("--json", required=True, help="output JSON path")
    score_parser.add_argument("--csv", required=True, help="output CSV path")
    score_parser.add_argument(
        "--atom-types",
        default="N",
        help=(
            "comma-separated atom selector; examples: N, N,O, NZ, SER:OG, 40, ALL, * "
            "(default: N)"
        ),
    )
    score_parser.add_argument(
        "--hydrogen-mode",
        choices=("auto", "explicit", "pdbfixer"),
        default="auto",
        help=(
            "auto uses explicit donor hydrogens if present, otherwise PDBFixer; "
            "explicit never invokes PDBFixer; pdbfixer always invokes PDBFixer"
        ),
    )
    score_parser.add_argument("--ph", type=float, default=7.0, help="pH for PDBFixer hydrogenation")
    score_parser.add_argument(
        "--hydrogen-minimize",
        choices=tuple(sorted(HYDROGEN_MINIMIZE_MODES)),
        default=HYDROGEN_MINIMIZE_AUTO,
        help=(
            "when PDBFixer adds hydrogens, attempt restrained OpenMM minimization, "
            "require it, or skip it (default: auto)"
        ),
    )
    score_parser.add_argument(
        "--hydrogen-forcefield",
        choices=tuple(sorted(HYDROGEN_FORCEFIELD_MODES)),
        default=HYDROGEN_FORCEFIELD_AUTO,
        help=(
            "force field backend for hydrogen minimization; auto tries Amber, "
            "then CHARMM if needed (default: auto)"
        ),
    )
    score_parser.add_argument(
        "--ccd-cache",
        default=None,
        help="directory for cached RCSB CCD CIF files used in CHARMM diagnostics",
    )
    score_parser.add_argument(
        "--ccd-online",
        choices=tuple(sorted(CCD_ONLINE_MODES)),
        default=CCD_ONLINE_AUTO,
        help="whether CHARMM diagnostics may fetch missing CCD files (default: auto)",
    )
    score_parser.add_argument(
        "--context-mode",
        choices=(CONTEXT_MODE_FAST, CONTEXT_MODE_NONE),
        default=CONTEXT_MODE_FAST,
        help=(
            "fast adds local solvent/packing/electrostatic/hydrophobic context fields; "
            "none leaves context fields empty (default: fast)"
        ),
    )
    score_parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="number of CPU worker processes for scoring/context work (default: 1)",
    )
    score_parser.add_argument(
        "--hbond-distance-cutoff",
        type=_positive_float,
        default=3.5,
        help="maximum donor-acceptor heavy atom distance in Angstrom before scoring (default: 3.5)",
    )
    score_parser.add_argument(
        "--hbond-per-donor-hydrogen",
        choices=tuple(sorted(HBOND_PER_DONOR_HYDROGEN_MODES)),
        default=None,
        help=(
            "reduce output to one HBond per donor hydrogen using the selected criterion; "
            "omitted reports all scored HBonds"
        ),
    )

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="write a hydrogenated PDB with PDBFixer/OpenMM",
    )
    prepare_parser.add_argument("input_pdb", help="input PDB file")
    prepare_parser.add_argument("--output", required=True, help="hydrogenated PDB output path")
    prepare_parser.add_argument("--ph", type=float, default=7.0, help="pH for PDBFixer hydrogenation")
    prepare_parser.add_argument(
        "--hydrogen-minimize",
        choices=tuple(sorted(HYDROGEN_MINIMIZE_MODES)),
        default=HYDROGEN_MINIMIZE_AUTO,
        help="attempt, require, or skip restrained OpenMM minimization (default: auto)",
    )
    prepare_parser.add_argument(
        "--hydrogen-forcefield",
        choices=tuple(sorted(HYDROGEN_FORCEFIELD_MODES)),
        default=HYDROGEN_FORCEFIELD_AUTO,
        help=(
            "force field backend for hydrogen minimization; auto tries Amber, "
            "then CHARMM if needed (default: auto)"
        ),
    )
    prepare_parser.add_argument(
        "--ccd-cache",
        default=None,
        help="directory for cached RCSB CCD CIF files used in CHARMM diagnostics",
    )
    prepare_parser.add_argument(
        "--ccd-online",
        choices=tuple(sorted(CCD_ONLINE_MODES)),
        default=CCD_ONLINE_AUTO,
        help="whether CHARMM diagnostics may fetch missing CCD files (default: auto)",
    )

    batch_parser = subparsers.add_parser("batch", help="batch scoring stub")
    batch_parser.add_argument("batch_input", help="reserved for a future batch format")

    return parser


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed
