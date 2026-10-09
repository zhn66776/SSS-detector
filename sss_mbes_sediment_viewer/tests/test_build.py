import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
import numpy as np
from build import (ELEVATION_RAMP, TILE, box_sum, check_elevation_like, fill_grid, grid_spec, hillshade, hillshade_window,
                   ramp_lut, reduce2, rough, scan, write_pyramid)


class BuildTests(unittest.TestCase):
    def test_box_sum_matches_brute_force(self):
        rng = np.random.default_rng(0)
        a = rng.random((17, 23))
        k, h = 5, 2
        p = np.pad(a, h)
        expect = np.array([[p[r:r + k, c:c + k].sum() for c in range(a.shape[1])] for r in range(a.shape[0])])
        self.assertTrue(np.allclose(box_sum(a, k), expect))

    def test_flat_surface_is_low_roughness(self):
        out = rough(np.ones((21, 21), np.float32))
        self.assertLess(np.nanmax(out), 1e-6)

    def test_feature_increases_roughness(self):
        z = np.ones((21, 21), np.float32)
        z[10, 10] = 2
        self.assertGreater(rough(z)[10, 10], .5)

    def test_banded_roughness_equals_single_band(self):
        rng = np.random.default_rng(1)
        z = rng.normal(-30, 1, (300, 40)).astype(np.float32)
        z[50:70, 5:9] = np.nan
        a, b = rough(z, band=64), rough(z, band=10000)
        self.assertTrue(np.array_equal(np.isnan(a), np.isnan(b)))
        self.assertTrue(np.allclose(a[np.isfinite(a)], b[np.isfinite(b)], atol=1e-5))

    def test_roughness_needs_half_the_window_valid(self):
        z = np.ones((21, 21), np.float32)
        out = rough(z)
        self.assertTrue(np.isfinite(out[10, 10]))
        self.assertTrue(np.isnan(out[0, 20]))  # grid corner: 36 of 121 cells
        island = np.full((21, 21), np.nan, np.float32)
        island[9:12, 9:12] = 1
        self.assertTrue(np.isnan(rough(island)).all())  # 9 of 121 cells

    def test_reduce2_is_nan_aware_and_handles_odd_shapes(self):
        g = np.full((5, 7), np.nan, np.float32)
        g[0, 0], g[0, 1], g[1, 0] = 1, 2, 3
        g[4, 6] = 8
        r = reduce2(g, band=2)
        self.assertEqual(r.shape, (3, 4))
        self.assertAlmostEqual(float(r[0, 0]), 2.0)
        self.assertEqual(r[2, 3], 8)
        self.assertTrue(np.isnan(r[1, 1]))

    def test_scan_flags_integer_count_column(self):
        pts = [(479005.75, 6548188.25, 2.0), (479006.25, 6548188.25, 3.0), (479006.75, 6548188.75, 3.0)]
        summary = scan(iter(pts))
        self.assertEqual(summary['integer_valued_fraction'], 1)
        self.assertEqual(summary['bounds'], [479005.75, 6548188.25, 479006.75, 6548188.75])
        with self.assertRaises(ValueError):
            check_elevation_like(summary, 'fake_0.5m')
        check_elevation_like(scan(iter([(0, 0, -29.695), (2, 0, -29.7)])), 'fake_2m')

    def test_grid_is_cell_centred_north_up(self):
        spec = grid_spec([478988.0, 6548189.0, 478992.0, 6548191.0], 2.0)
        self.assertEqual((spec['left'], spec['top'], spec['width'], spec['height']), (478987.0, 6548192.0, 3, 2))
        pts = [(478988.0, 6548189.0, -30.0), (478992.0, 6548191.0, -20.0), (478992.0, 6548191.0, -21.0)]
        grid, kept, dup = fill_grid(iter(pts), spec)
        self.assertEqual((kept, dup), (3, 1))
        self.assertEqual(grid[1, 0], -30.0)
        self.assertEqual(grid[0, 2], -21.0)
        self.assertTrue(np.isnan(grid[0, 0]))

    def test_hillshade_lights_from_the_north_west(self):
        y, x = np.mgrid[0:50, 0:50].astype(float)
        faces_nw = 0.2 * x + 0.2 * y  # descends toward the west and the north
        self.assertGreater(hillshade(faces_nw, 2.0)[10:40, 10:40].mean(), 0.9)
        self.assertLess(hillshade(-faces_nw, 2.0)[10:40, 10:40].mean(), 0.4)
        self.assertAlmostEqual(float(hillshade(np.zeros((5, 5)), 2.0)[2, 2]), np.sin(np.radians(45)), places=5)

    def test_hillshade_window_matches_whole_grid_at_every_tile(self):
        rng = np.random.default_rng(2)
        level = rng.normal(-30, .3, (TILE * 2 + 40, TILE * 2 + 40)).astype(np.float32)
        whole = hillshade(level, 2.0)
        for r0, c0 in [(0, 0), (TILE, 0), (0, TILE), (TILE, TILE), (2 * TILE, 2 * TILE)]:
            w = hillshade_window(level, r0, c0, 2.0)
            block = whole[r0:r0 + TILE, c0:c0 + TILE]
            self.assertEqual(w.shape, (TILE, TILE))
            # interior cells (away from the grid edge) shade identically with or without the halo
            self.assertTrue(np.allclose(w[1:block.shape[0] - 1, 1:block.shape[1] - 1], block[1:-1, 1:-1], atol=1e-5))

    def test_pyramid_levels_and_present_tiles(self):
        grid = np.full((300, 600), np.nan, np.float32)
        grid[:200, :200] = -30.0
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            info = write_pyramid(grid, out, -31, -29, ramp_lut(ELEVATION_RAMP))
            self.assertEqual(info['max_zoom'], 2)
            self.assertEqual(sorted(info['present']['2']), ['0_0'])
            self.assertEqual(info['present']['0'], ['0_0'])
            self.assertTrue((out / '2' / '0_0.png').exists())
            self.assertFalse((out / '2' / '2_1.png').exists())


if __name__ == '__main__':
    unittest.main()
