"""Command line interface for backbone amide HBond scoring."""

from __future__ import annotations

import argparse
import sys

from .backbone_amide import score_pdb, write_csv, write_json
from .hydrogen import add_hydrogens_with_pdbfixer


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "score":
            result = score_pdb(args.input_pdb, hydrogen_mode=args.hydrogen_mode, ph=args.ph)
            write_json(result, args.json)
            write_csv(result, args.csv)
            print(
                f"Scored {result.counts['hbonds']} HBonds "
                f"from {result.counts['backbone_donors']} donors and "
                f"{result.counts['backbone_acceptors']} acceptors."
            )
            print(
                "Timing seconds: "
                f"parse={result.timing_seconds['parse']:.6f}, "
                f"hydrogen={result.timing_seconds['hydrogen']:.6f}, "
                f"score={result.timing_seconds['score']:.6f}, "
                f"total={result.timing_seconds['total']:.6f}"
            )
            return 0
        if args.command == "prepare":
            add_hydrogens_with_pdbfixer(args.input_pdb, args.output, ph=args.ph)
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
        description="Score backbone amide HBonds with ChemEM HBond scoring functions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="score one protein PDB")
    score_parser.add_argument("input_pdb", help="input PDB file")
    score_parser.add_argument("--json", required=True, help="output JSON path")
    score_parser.add_argument("--csv", required=True, help="output CSV path")
    score_parser.add_argument(
        "--hydrogen-mode",
        choices=("auto", "explicit", "pdbfixer"),
        default="auto",
        help=(
            "auto uses explicit backbone hydrogens if present, otherwise PDBFixer; "
            "explicit never invokes PDBFixer; pdbfixer always invokes PDBFixer"
        ),
    )
    score_parser.add_argument("--ph", type=float, default=7.0, help="pH for PDBFixer hydrogenation")

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="write a hydrogenated PDB with PDBFixer/OpenMM",
    )
    prepare_parser.add_argument("input_pdb", help="input PDB file")
    prepare_parser.add_argument("--output", required=True, help="hydrogenated PDB output path")
    prepare_parser.add_argument("--ph", type=float, default=7.0, help="pH for PDBFixer hydrogenation")

    batch_parser = subparsers.add_parser("batch", help="batch scoring stub")
    batch_parser.add_argument("batch_input", help="reserved for a future batch format")

    return parser
