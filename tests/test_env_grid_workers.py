from __future__ import annotations

import unittest

try:
    import numpy as np
    from env_grids import site_map_utils
except Exception as exc:  # pragma: no cover - depends on optional ChemEM stack
    np = None
    site_map_utils = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(site_map_utils is None, f"env grid dependencies unavailable: {IMPORT_ERROR}")
class EnvGridWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        required = [
            "make_protein_and_solvent_masks_cpp",
            "propagate_logp_exp_decay_cpp",
            "compute_electrostatic_grid_cutoff_cpp",
        ]
        missing = [name for name in required if not hasattr(site_map_utils.grid_maps, name)]
        if missing:
            raise unittest.SkipTest(f"grid_maps extension missing: {', '.join(missing)}")

    def test_mask_grid_workers_match_serial(self) -> None:
        coords = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]], dtype=float)
        radii = np.array([1.0, 1.2], dtype=float)
        origin = np.array([0.0, 0.0, 0.0], dtype=float)
        shape = (6, 6, 6)

        serial = site_map_utils.make_protein_and_solvent_masks(
            coords,
            radii,
            origin,
            shape,
            1.0,
            workers=1,
        )
        parallel = site_map_utils.make_protein_and_solvent_masks(
            coords,
            radii,
            origin,
            shape,
            1.0,
            workers=2,
        )

        for serial_grid, parallel_grid in zip(serial, parallel):
            np.testing.assert_allclose(serial_grid, parallel_grid)

    def test_hydrophobic_decay_workers_match_serial(self) -> None:
        centers = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]], dtype=float)
        values = np.array([0.8, -0.2], dtype=float)
        origin = np.array([0.0, 0.0, 0.0], dtype=float)
        mask = np.ones((5, 5, 5), dtype=np.uint8)

        serial = site_map_utils.propagate_logp_exp_decay_cpp(
            centers,
            values,
            origin,
            mask.shape,
            1.0,
            sasa_mask=mask,
            workers=1,
        )
        parallel = site_map_utils.propagate_logp_exp_decay_cpp(
            centers,
            values,
            origin,
            mask.shape,
            1.0,
            sasa_mask=mask,
            workers=2,
        )

        np.testing.assert_allclose(serial, parallel)

    def test_electrostatic_grid_workers_match_serial(self) -> None:
        positions = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]], dtype=float)
        charges = np.array([0.5, -0.5], dtype=float)
        origin = np.array([0.0, 0.0, 0.0], dtype=float)
        apix = np.array([1.0, 1.0, 1.0], dtype=float)
        shape = (5, 5, 5)

        serial = site_map_utils.compute_electrostatic_grid_cutoff_cpp(
            positions,
            charges,
            shape,
            origin,
            apix,
            workers=1,
        )
        parallel = site_map_utils.compute_electrostatic_grid_cutoff_cpp(
            positions,
            charges,
            shape,
            origin,
            apix,
            workers=2,
        )

        np.testing.assert_allclose(serial, parallel)


if __name__ == "__main__":
    unittest.main()
