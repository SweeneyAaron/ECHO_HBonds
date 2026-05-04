# HBOND_CHEMEM

This folder now contains two related pieces:

- the new backbone amide HBond CLI/package implemented in `backbone_amide.py`;
- the original `reference_hbond_score.py`, which remains a dependency-free
  reference for ChemEM's docking HBond branch.

For installation, CLI usage, JSON/CSV fields, timing notes, and the uncapped
backbone amide scorer, see the top-level `README.md`.

This folder is a standalone reference for reproducing the hydrogen-bond part of the ChemEM docking score. It captures the data and logic needed to go from atom typing, to donor/acceptor pair identification, to the angle-dependent HBond Buckingham-style energy used by the active docking path.

## Files

- `data/HBOND_POLY_A.json`, `data/HBOND_POLY_B.json`, `data/HBOND_POLY_C.json`: angle-dependent polynomial coefficients for HBond `A`, `B`, and `C` terms.
- `data/TableA.json`, `data/TableB.json`, `data/TableC.json`: base atom-type pair Buckingham parameters used by the nonbonded score.
- `data/hbond_roles.json`: donor/acceptor type IDs copied from `ChemEM/data.py`, plus the donor/acceptor keys present in the polynomial tables.
- `data/atom_type_summary.tsv`: generated summary of ChemEM atom type IDs and whether each appears in HBond role lists and polynomial tables.
- `reference_hbond_score.py`: dependency-free Python reference for scoring one HBond pair.

## Active Code Path

The active docking route is:

1. Ligands are loaded in `ChemEM/parsers.py`.
2. Ligand heavy atoms are typed with `AtomType.from_atom(...)`.
3. Protein heavy atoms are typed in `ChemEM/scoring_functions/precompute_data.py` with `AtomType.from_id(res.name, atom.name)`.
4. `PreCompDataLigand + PreCompDataProtein2` build the combined precomputed object.
5. `ChemEM/protocols/docking.py` calls `docking.run_aco(...)`.
6. The C++ ACO path uses `PrecomputedDataCPP2` and scores poses with `echo_score_v2(...)` in `ChemEM/cpp/docking/echo_score.cpp`.

The older `docking.run_echo_score(...)` path uses `PrecomputedDataCPP` and `echo_score_full(...)`; it is still present and used for some rescoring/debug paths, but the ACO docking score target here is `echo_score_v2`.

## Atom Typing

Ligand typing comes from `ChemEM/data.py::AtomType.from_atom`:

- special high-priority ligand patterns are checked first: peptide O/N, amide O/N, phosphate double-bonded O, and phosphate donor O;
- fallback matching uses element symbol, RDKit bond types, explicit hydrogen count, and aromaticity;
- only non-hydrogen atoms are typed for docking.

Protein typing comes from `ChemEM/data.py::AtomType.from_id`:

- residue name and PDB atom name are looked up in `protein_atom_data`;
- this assigns protein-specific types such as `PEPTIDE_O`, `PEPTIDE_N`, `NITROGEN_ASNGLN`, and `NITROGEN_TRPNE1`.

Protein donor/acceptor role for the C++ direction choice is not inferred from the polynomial table. It is built by `get_role_int(res.name, atom.name)` from `PROTEIN_DONOR_ATOM_IDS` and `PROTEIN_ACCEPTOR_ATOM_IDS`.

## Donor And Acceptor Identification

Pair eligibility is first computed in `compute_donor_acceptor_mask(...)`:

```text
mask[protein_i, ligand_j] = (
    protein_type in HBOND_DONOR_ATOM_IDXS
    and ligand_type in HBOND_ACCEPTOR_ATOM_IDXS
) or (
    protein_type in HBOND_ACCEPTOR_ATOM_IDXS
    and ligand_type in HBOND_DONOR_ATOM_IDXS
)
```

The HBond IDs from `ChemEM/data.py` are:

```text
Donors:    [13, 15, 19, 23, 24, 28, 37, 40, 42, 43]
Acceptors: [13, 14, 15, 16, 17, 18, 19, 36, 20, 21, 22, 23, 24, 25, 26, 28, 38, 39, 41]
```

The polynomial tables contain:

```text
Polynomial donors:    [13, 15, 19, 23, 24, 28, 37, 40, 42, 43]
Polynomial acceptors: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 38, 39, 41]
```

Important discrepancy: `HBOND_ACCEPTOR_ATOM_IDXS` includes `36`, while the polynomial tables include `27`. Reproduce the exact code path you care about:

- `compute_donor_acceptor_mask(...)` uses the IDs from `ChemEM/data.py`;
- C++ polynomial lookup uses the donor and acceptor keys loaded from `HBOND_POLY_A/B/C.json`;
- if a pair passes the mask but has no polynomial coefficients for the selected donor/acceptor direction, it cannot receive an HBond polynomial contribution.

## HBond Geometry And Energy

In `echo_score_v2`, the HBond branch runs inside the ligand/protein nonbonded loop:

1. Skip if the heavy-atom distance is `>= 6.0 A`.
2. Skip if `hbond_donor_acceptor_mask[protein_i, ligand_j]` is false.
3. Build candidate directions:
   - protein donor to ligand acceptor, if protein role is donor/both and ligand polynomial role is acceptor/both;
   - ligand donor to protein acceptor, if protein role is acceptor/both and ligand polynomial role is donor/both.
4. For each candidate direction, compute donor-H-acceptor angles and keep the maximum angle.
5. Use the direction with the largest angle.
6. Require `best_angle > 110.0` degrees.
7. Look up coefficients with donor type first and acceptor type second.
8. Evaluate `A(angle)`, `B(angle)`, and `C(angle)` using Horner's method.
9. Clamp distance with `r_clamped = max(r, 2.0)`.
10. Compute:

```text
raw_hbond = A(angle) * exp(-B(angle) * r_clamped) - C(angle) / r_clamped^6
```

11. If `raw_hbond < 0`, multiply it by the environment scale sampled at the ligand atom position.
12. If the value is positive, cap it:

```text
contribution = repCap * tanh(raw_hbond / repCap)
```

Otherwise the contribution is the negative value unchanged after environment scaling.

The HBond contribution is accumulated in the `nonbond` bucket. In the final `echo_score_v2` score, it is weighted as part of:

```text
w_nonbond * (nonbond + aromatic + halogen_bond_score)
```

Current `PreCompDataProtein2` defaults include `w_nonbond = 0.013767`.

## Reference Script

Run the built-in checks:

```bash
python3 HBOND_CHEMEM/reference_hbond_score.py
```

Use `score_hbond_pair(...)` for a ligand/protein pair:

```python
from HBOND_CHEMEM.reference_hbond_score import ACCEPTOR_BIT, score_hbond_pair

score = score_hbond_pair(
    ligand_atom_type=13,
    protein_atom_type=19,
    ligand_pos=(0.0, 0.0, 0.0),
    ligand_h_positions=[(1.0, 0.0, 0.0)],
    protein_pos=(2.8, 0.0, 0.0),
    protein_role=ACCEPTOR_BIT,
    env_scale=1.0,
)

print(score.contribution)
```

For exact active-path reproduction, pass the protein role generated from residue and atom name by ChemEM's `get_role_int(...)`. If `protein_role` is omitted, the reference script infers it from atom type IDs, which is useful for synthetic examples but less exact than the docking code.
