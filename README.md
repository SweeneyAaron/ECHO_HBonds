# Protein HBond Scoring

This repository scores protein hydrogen bonds in PDB files with the ChemEM
HBond polynomial scoring functions. It writes one JSON file and one CSV file
containing atom identifiers, HBond geometry, uncapped ChemEM HBond scores, and
fast local environment descriptors for downstream HDX correlation analysis.

By default, the scorer reports HBonds where an atom named `N` participates as
either donor or acceptor. Bare selector tokens are strict PDB atom-name matches,
not element matches, so `N` means atom name `N` rather than every nitrogen atom.
The selector can be expanded to other PDB atom names, residue-qualified atom
names, ChemEM type IDs, or all detected protein HBonds. Waters and ligands are
ignored. There is no sequence-separation or residue-locality filter: every typed
donor/acceptor pair can score if it passes the ChemEM distance and angle
cutoffs.

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

Score a PDB that already contains donor hydrogens:

```bash
python -m HBOND_CHEMEM score input_with_hydrogens.pdb --atom-types N --hydrogen-mode explicit --json hbonds.json --csv hbonds.csv
```

Score a PDB and let the program add hydrogens with PDBFixer if none are present:

```bash
python -m HBOND_CHEMEM score test_data/2erk.pdb --atom-types N --hydrogen-mode auto --json 2erk_hbonds.json --csv 2erk_hbonds.csv
```

When PDBFixer adds hydrogens, the default `--hydrogen-minimize auto` attempts a
short OpenMM minimization with heavy atoms restrained and hydrogens free to
relax. It tries Amber first, then CHARMM 2024 if Amber cannot parameterize the
structure. To force CHARMM for modified residues such as phosphorylated `TPO` or
`PTR`:

```bash
python -m HBOND_CHEMEM score test_data/2erk.pdb --hydrogen-forcefield charmm --json 2erk_hbonds.json --csv 2erk_hbonds.csv
```

The CHARMM path can add hydrogens for modified residues that already have
installed CHARMM templates, including `TPO`, `PTR`, and `SEP`. RCSB/wwPDB CCD
CIF files are used only for diagnostics/cache metadata for unsupported
heterologous residues; the force-field parameters still need to come from
OpenMM's installed force-field XML files. To skip minimization entirely:

```bash
python -m HBOND_CHEMEM score test_data/2erk.pdb --hydrogen-minimize none --json 2erk_hbonds.json --csv 2erk_hbonds.csv
```

Use `--hydrogen-minimize restrained` when unsupported residues should be a hard
error instead of an automatic no-minimization fallback. Use `--ccd-cache` and
`--ccd-online never` to force offline CCD diagnostics from an existing cache.

Selector examples:

```bash
python -m HBOND_CHEMEM score input.pdb --atom-types N,O --json hbonds.json --csv hbonds.csv
python -m HBOND_CHEMEM score input.pdb --atom-types SER:OG,LYS:NZ --json hbonds.json --csv hbonds.csv
python -m HBOND_CHEMEM score input.pdb --atom-types 40,39 --json hbonds.json --csv hbonds.csv
python -m HBOND_CHEMEM score input.pdb --atom-types ALL --json hbonds.json --csv hbonds.csv
```

Environment context fields are written by default. To write the same HBond rows
with empty context fields, use:

```bash
python -m HBOND_CHEMEM score input.pdb --context-mode none --json hbonds.json --csv hbonds.csv
```

For larger structures, use multiple CPU processes for the Python scoring and
fast-context loops:

```bash
python -m HBOND_CHEMEM score input.pdb --workers 4 --json hbonds.json --csv hbonds.csv
```

The default is `--workers 1`, which keeps the serial execution path. Worker
requests above the available CPU count are accepted and capped internally; the
JSON metadata records the requested worker count and the effective score/context
worker counts used for the run.

The donor-acceptor heavy-atom distance cutoff defaults to `3.5 A`. Candidate
pairs at or above this distance are skipped before angle and score evaluation.
To relax or tighten it:

```bash
python -m HBOND_CHEMEM score input.pdb --hbond-distance-cutoff 4.0 --json hbonds.json --csv hbonds.csv
```

To reduce the output to one HBond per donor hydrogen, choose the selection
criterion:

```bash
python -m HBOND_CHEMEM score input.pdb --hbond-per-donor-hydrogen best-distance --json hbonds.json --csv hbonds.csv
python -m HBOND_CHEMEM score input.pdb --hbond-per-donor-hydrogen best-normalized-score --json hbonds.json --csv hbonds.csv
```

`best-distance` keeps the lowest donor-acceptor heavy-atom distance. If there is
a tie, it prefers higher `normalized_score`, lower hydrogen-acceptor distance,
and then higher angle before falling back to stable atom identity. Omitting the
flag preserves the full one-row-per-scored-HBond output.

Prepare a hydrogenated PDB first, then run the fast explicit-H scoring path:

```bash
python -m HBOND_CHEMEM prepare test_data/2erk.pdb --output 2erk_h.pdb --ph 7.0 --hydrogen-forcefield charmm
python -m HBOND_CHEMEM score 2erk_h.pdb --atom-types N --hydrogen-mode explicit --json 2erk_hbonds.json --csv 2erk_hbonds.csv
```

Use `--hydrogen-minimize none` on `prepare` when you only want PDBFixer-added
hydrogens without the OpenMM minimization step.

The installed console script is equivalent:

```bash
hbond-chemem score test_data/2erk.pdb --json 2erk_hbonds.json --csv 2erk_hbonds.csv
```

Batch mode is intentionally a stub until the batch input format is finalized:

```bash
python -m HBOND_CHEMEM batch batch_input_placeholder
```

## Scoring Rules

The program identifies common protein donors and acceptors:

- backbone `N` donors and backbone `O` acceptors;
- side-chain donors/acceptors for common ASN/GLN, ARG, LYS, HIS, TRP, SER,
  THR, TYR, CYS, ASP, GLU, and MET atoms where the role is chemically feasible;
- donor hydrogens only when present explicitly or added by PDBFixer.

The compact ChemEM type map includes backbone `N = 40`, backbone `O = 39`,
amide side-chain `N = 43`, amide side-chain `O = 38`, generic/charged
`O = 19`, aromatic/generic nitrogen types, and `S = 24`.

`--atom-types` filters scored HBonds after donor/acceptor typing. A row is
reported when the selected atom participates on either side of the HBond. Bare
text selectors are strict PDB atom-name matches. Examples:

- `N`: all HBonds involving atoms named `N`;
- `N,O`: all HBonds involving atoms named `N` or `O`;
- `NZ` or `OG`: all HBonds involving those PDB atom names;
- `SER:OG`: all HBonds involving that residue-qualified atom name;
- `40`: all HBonds involving ChemEM type ID `40`;
- `ALL` or `*`: no atom filter.

For each candidate pair, the scorer requires:

- donor-acceptor heavy atom distance below `--hbond-distance-cutoff`
  (`3.5 A` by default);
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
`HBOND_CHEMEM/data/hbond_score_bounds.json`.

## Environment Context

By default, `--context-mode fast` annotates each scored HBond with dependency-light
local descriptors sampled directly around the donor hydrogen and the HBond
midpoint. This is a fast approximation inspired by the heavier ChemEM
environment-grid ideas; it does not build full grids and does not require
NumPy, SciPy, ChemEM, or compiled grid extensions.

The new fields are intended for empirical correlation with HDX experiments, not
as a calibrated protection-factor model:

- `env_h_sasa_fraction`: Shrake-Rupley-style exposure fraction around the donor
  hydrogen;
- `env_h_solvent_reach_fraction`: ray-based solvent-reach proxy from the donor
  hydrogen;
- `env_h_packing_count_6p5`: heavy-atom packing count within 6.5 A of the donor
  hydrogen;
- `env_h_electrostatic`: local Coulomb-like formal/polar charge proxy;
- `env_h_hydrophobic`: local residue/element hydrophobicity proxy;
- `env_mid_sasa_fraction`, `env_mid_electrostatic`, and
  `env_mid_hydrophobic`: the corresponding midpoint context values.

The donor atom, donor hydrogen, and acceptor atom for the scored HBond are
excluded from the context calculations so the values describe the surrounding
environment rather than the HBond itself. The JSON metadata records the context
mode, constants, proxy models, and context timing for each run.

## Outputs

The JSON file contains:

- `metadata`: input path, hydrogen mode/source, atom selector, cutoffs, ChemEM
  donor/acceptor types present in the scored rows, context settings, counts,
  raw HBond candidate count before optional per-hydrogen selection, hydrogen
  minimization settings, selected force field, CHARMM/CCD diagnostics, version
  metadata, timing, `normalization_mode`, and `rep_cap_removed`;
- `hbonds`: one object per scored HBond.

The CSV has the same one-row-per-HBond records. Important fields include:

- atom ids: `donor_atom_id`, `donor_h_atom_id`, `acceptor_atom_id`;
- residue ids: chain, residue name, residue number, insertion code for donor
  and acceptor;
- geometry: donor, hydrogen, and acceptor coordinates, donor-acceptor distance,
  hydrogen-acceptor distance, and donor-H-acceptor angle;
- scoring values: donor/acceptor ChemEM types, evaluated `a_value`, `b_value`,
  `c_value`, `hbond_score`, and `normalized_score`;
- environment context values: the `env_h_*` and `env_mid_*` fields described
  above.

If PDBFixer generates a hydrogen, its id is stable and synthetic, for example:

```text
generated:model:1:chain:A:42:_:H
```

Original heavy atom serial numbers from the input PDB are preserved for donor
and acceptor atoms whenever they can be mapped by chain, residue, and atom name.

### ChimeraX per-residue attribute files

Pass `--chimerax <output_dir>` on the `score` subcommand to additionally write a
directory of ChimeraX `defattr` attribute files for per-residue visualisation,
similar to colouring by B-factor:

```bash
python -m HBOND_CHEMEM score 2erk_h.pdb --hydrogen-mode explicit \
    --json 2erk_hbonds.json --csv 2erk_hbonds.csv \
    --chimerax 2erk_chimerax
```

For each residue that participates (as donor or acceptor) in at least one
scored HBond, the residue's per-attribute value is taken from the single bond
touching it with the highest `normalized_score`. Residues with no scored bond
are omitted.

The directory contains one `.defattr` file per visualisable attribute plus a
`hbond_chimerax.cxc` helper script:

- `hbond_score.defattr` → ChimeraX attribute `hbondScore`
- `normalized_score.defattr` → `normalizedScore`
- `env_h_sasa_fraction.defattr` → `envHSasaFraction`
- `env_h_solvent_reach_fraction.defattr` → `envHSolventReachFraction`
- `env_h_packing_count_6p5.defattr` → `envHPackingCount6p5`
- `env_h_electrostatic.defattr` → `envHElectrostatic`
- `env_h_hydrophobic.defattr` → `envHHydrophobic`
- `env_mid_sasa_fraction.defattr` → `envMidSasaFraction`
- `env_mid_electrostatic.defattr` → `envMidElectrostatic`
- `env_mid_hydrophobic.defattr` → `envMidHydrophobic`

Under `--context-mode none` the eight `env_*` files are skipped and only the
two score files plus the `.cxc` helper are written.

To visualise in ChimeraX, open the PDB and then run the helper script from its
output directory:

```text
open 2erk_h.pdb
cd 2erk_chimerax
open hbond_chimerax.cxc
```

The helper opens every defattr file and applies an example
`color byattribute r:hbondScore palette blue-white-red`. Swap `r:hbondScore`
for any of the other attribute names listed in the script's comments to colour
by a different attribute.

#### Fixing the colour range

By default, `color byattribute` maps the palette across the data's actual
min/max, so the same colour can mean different values across structures.
Append `range min,max` to pin the mapping:

```text
color byattribute r:hbondScore palette blue-white-red range -6,0
```

Sensible defaults:

- `r:hbondScore`: `range -6,0` (more negative is stronger; values near zero are weakly favourable);
- `r:normalizedScore`, `r:envHSasaFraction`, `r:envHSolventReachFraction`, `r:envMidSasaFraction`: `range 0,1`;
- packing/electrostatic/hydrophobic attributes: omit `range` and let ChimeraX use the data range, or set it from inspection of the CSV.

#### Adding a colour key

Use the `key` command with `colour:label` pairs to draw a legend in the
graphics window. Stops should match the palette and range:

```text
key blue:-6 white:-3 red:0  size 0.025,0.4  pos 0.9,0.3  fontSize 14
```

A typical paired block to drop into the `.cxc` (or run interactively) is:

```text
color byattribute r:hbondScore palette blue-white-red range -6,0
key blue:-6 white:-3 red:0  size 0.025,0.4  pos 0.9,0.3  fontSize 14
```

`pos x,y` is fractional screen position of the key's bottom-left corner;
`size w,h` is its fractional width and height. Re-run `key` with new stops
when you switch attributes so the legend tracks the active colouring.

## Timing

The approximate 1-second target applies to parsing, candidate search, scoring,
fast context annotation, and JSON/CSV output on a typical single protein.
`--workers` can speed up larger CPU-bound scoring/context workloads, although
small proteins may stay faster on the serial path because process startup has a
cost.
Hydrogenation with
PDBFixer/OpenMM is measured separately because it is a structure-preparation
step and may dominate cold-start runtime. For repeated runs, prepare hydrogens
once and score the hydrogenated PDB with `--hydrogen-mode explicit`.

## Tests

Run the dependency-light test suite:

```bash
python -m unittest discover -s tests
```
