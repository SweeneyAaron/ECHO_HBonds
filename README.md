# Backbone Amide HBond Scoring

This repository scores backbone amide hydrogen bonds in protein PDB files with
the ChemEM HBond polynomial scoring functions. It writes one JSON file and one
CSV file containing atom identifiers, HBond geometry, and uncapped ChemEM HBond
scores.

Only backbone peptide `N-H -> O` interactions are scored in v1. Side-chain
amides, waters, ligands, and terminal `OXT` atoms are ignored. There is no
sequence-separation or residue-locality filter: every backbone donor/acceptor
pair can score if it passes the ChemEM distance and angle cutoffs.

## Environment

The hydrogen-addition path uses PDBFixer/OpenMM, so use Python 3.12. The
recommended setup is conda with the `conda-forge` channel. From the repository
root, create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate hbond-chemem
```

That file already pins `conda-forge` as the package channel:

```yaml
channels:
  - conda-forge
```

After activating the environment, install this local package in editable mode:

```bash
pip install -e .
```

This makes both forms available:

```bash
python -m HBOND_CHEMEM --help
hbond-chemem --help
```

If you prefer not to use `environment.yml`, the equivalent explicit conda
command is:

```bash
conda create -n hbond-chemem -c conda-forge python=3.12 openmm pdbfixer pip
conda activate hbond-chemem
pip install -e .
```

Or install with pip in a Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The scorer itself is CPU-only. PDBFixer/OpenMM are used only when hydrogens are
missing or when you explicitly run the preparation command.

## Run

Score a PDB that already contains peptide hydrogens:

```bash
python -m HBOND_CHEMEM score input_with_hydrogens.pdb --hydrogen-mode explicit --json hbonds.json --csv hbonds.csv
```

Score a PDB and let the program add hydrogens with PDBFixer if none are present:

```bash
python -m HBOND_CHEMEM score test_data/2erk.pdb --hydrogen-mode auto --json 2erk_hbonds.json --csv 2erk_hbonds.csv
```

Prepare a hydrogenated PDB first, then run the fast explicit-H scoring path:

```bash
python -m HBOND_CHEMEM prepare test_data/2erk.pdb --output 2erk_h.pdb --ph 7.0
python -m HBOND_CHEMEM score 2erk_h.pdb --hydrogen-mode explicit --json 2erk_hbonds.json --csv 2erk_hbonds.csv
```

The installed console script is equivalent:

```bash
hbond-chemem score test_data/2erk.pdb --json 2erk_hbonds.json --csv 2erk_hbonds.csv
```

Batch mode is intentionally a stub until the batch input format is finalized:

```bash
python -m HBOND_CHEMEM batch batch_input_placeholder
```

## Scoring Rules

The program identifies:

- donors: protein `ATOM` records with backbone atom `N` and at least one peptide
  hydrogen named like `H`, `HN`, `H1`, `H2`, or `H3`;
- acceptors: protein `ATOM` records with backbone atom `O`;
- ChemEM atom types: donor `PEPTIDE_N = 40`, acceptor `PEPTIDE_O = 39`.

For each candidate pair, the scorer requires:

- donor-acceptor heavy atom distance `< 6.0 A`;
- best donor-H-acceptor angle `> 110.0 degrees`.

The score is:

```text
A(angle) * exp(-B(angle) * max(distance, 2.0)) - C(angle) / max(distance, 2.0)^6
```

`A`, `B`, and `C` are read from `HBOND_CHEMEM/data/HBOND_POLY_A.json`,
`HBOND_CHEMEM/data/HBOND_POLY_B.json`, and
`HBOND_CHEMEM/data/HBOND_POLY_C.json`. The old positive `repCap` is not applied;
positive and negative scores are emitted uncapped.

`normalized_score` converts the raw energy into a 0-1 favorable-strength score:

```text
normalized_score = clamp(-hbond_score / max_favorable_magnitude, 0, 1)
```

The maximum favorable magnitudes are precomputed for every ChemEM HBond
donor/acceptor table pair in
`HBOND_CHEMEM/data/hbond_score_bounds.json`. For the current backbone scorer,
the PEPTIDE_N `40` -> PEPTIDE_O `39` denominator is about `6.685645`.

## Outputs

The JSON file contains:

- `metadata`: input path, hydrogen mode/source, cutoffs, ChemEM atom types,
  counts, timing, `normalization_mode`, and `rep_cap_removed`;
- `hbonds`: one object per scored HBond.

The CSV has the same one-row-per-HBond records. Important fields include:

- atom ids: `donor_atom_id`, `donor_h_atom_id`, `acceptor_atom_id`;
- residue ids: chain, residue name, residue number, insertion code for donor
  and acceptor;
- geometry: donor, hydrogen, and acceptor coordinates, donor-acceptor distance,
  hydrogen-acceptor distance, and donor-H-acceptor angle;
- scoring values: donor/acceptor ChemEM types, evaluated `a_value`, `b_value`,
  `c_value`, `hbond_score`, and `normalized_score`.

If PDBFixer generates a hydrogen, its id is stable and synthetic, for example:

```text
generated:model:1:chain:A:42:_:H
```

Original heavy atom serial numbers from the input PDB are preserved for donor
`N` and acceptor `O` atoms whenever they can be mapped by chain, residue, and
atom name.

## Timing

The approximate 1-second target applies to parsing, candidate search, scoring,
and JSON/CSV output on a typical single protein. Hydrogenation with
PDBFixer/OpenMM is measured separately because it is a structure-preparation
step and may dominate cold-start runtime. For repeated runs, prepare hydrogens
once and score the hydrogenated PDB with `--hydrogen-mode explicit`.

## Tests

Run the dependency-light test suite:

```bash
python -m unittest discover -s tests
```
