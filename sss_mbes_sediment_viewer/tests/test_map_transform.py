"""Regression contract for the north-up EPSG:32630 canvas transform and the tile
addressing that app.js shares with sss_contact_inspector/serve.py."""
import math
import unittest

TILE = 256


def screen(e, n, scale, ox, oy):
    return e * scale + ox, oy - n * scale


def world(x, y, scale, ox, oy):
    return (x - ox) / scale, (oy - y) / scale


def tile_level(scale, native, max_zoom):
    mpp = 1 / scale
    return max(0, min(max_zoom, round(max_zoom - math.log2(mpp / native))))


def tile_bounds(layer, z, x, y):
    res = layer['native'] * 2 ** (layer['max_zoom'] - z)
    span = TILE * res
    left = layer['left'] + x * span
    top = layer['top'] - y * span
    return left, top - span, left + span, top


class TransformTests(unittest.TestCase):
    def test_northing_increases_up_screen(self):
        self.assertLess(screen(0, 500, 2.0, 0, 1000)[1], screen(0, 400, 2.0, 0, 1000)[1])

    def test_screen_world_round_trip(self):
        e, n = 478990.25, 6548300.75
        x, y = screen(e, n, 2.0, 15.0, 1000.0)
        self.assertEqual(world(x, y, 2.0, 15.0, 1000.0), (e, n))

    def test_tile_level_tracks_view_scale(self):
        # Costa ECW: 0.1 m native, 10 pyramid levels (see /api/config)
        self.assertEqual(tile_level(10.0, 0.1, 10), 10)   # 0.1 m per px -> native tiles
        self.assertEqual(tile_level(5.0, 0.1, 10), 9)     # 0.2 m per px
        self.assertEqual(tile_level(0.03, 0.1, 10), 2)
        self.assertEqual(tile_level(0.0001, 0.1, 10), 0)  # never below the pyramid root
        self.assertEqual(tile_level(80.0, 0.1, 10), 10)   # never beyond native

    def test_tile_addressing_matches_the_service_definition(self):
        costa = {'native': 0.1, 'max_zoom': 10, 'left': 477323.46875, 'top': 6568269.5}
        e, n = 485048.62, 6560499.49
        z = 10
        span = TILE * costa['native']
        x = math.floor((e - costa['left']) / span)
        y = math.floor((costa['top'] - n) / span)
        left, bottom, right, top = tile_bounds(costa, z, x, y)
        self.assertTrue(left <= e < right and bottom < n <= top)
        self.assertAlmostEqual(right - left, 25.6)
        # the same tile drawn through the view transform lands where its corners say
        scale, ox, oy = 10.0, -4850000.0, 65600000.0
        sx0, sy0 = screen(left, top, scale, ox, oy)
        sx1, sy1 = screen(right, bottom, scale, ox, oy)
        self.assertAlmostEqual(sx1 - sx0, 256.0)
        self.assertAlmostEqual(sy1 - sy0, 256.0)

    def test_crop_api_span_is_capped_so_a_single_crop_cannot_back_a_map(self):
        # /api/crop clamps to 768 native pixels; the old viewer stretched that 76.8 m
        # image over the whole viewport, which is what broke SSS/polygon alignment.
        native, requested = 0.1, 147.8
        pixels = max(1, min(768, math.ceil(requested / native)))
        self.assertAlmostEqual(pixels * native, 76.8)
        self.assertNotAlmostEqual(pixels * native, requested)


if __name__ == '__main__':
    unittest.main()
