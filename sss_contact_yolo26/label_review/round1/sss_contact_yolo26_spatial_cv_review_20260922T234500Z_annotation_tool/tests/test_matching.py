"""Tests for one-to-one matching and label formatting (in-memory only).

Run:  python3 -m unittest discover -s tests -v   (from the platform directory)
"""
import itertools
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sss_tool.labels import LabelError, format_boxes, parse_label_text, validate_box  # noqa: E402
from sss_tool.matching import box_iou, match_boxes  # noqa: E402


def B(x0, y0, x1, y1):
    """Box from corners in normalised coordinates."""
    return {"cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2, "w": x1 - x0, "h": y1 - y0}


def brute_force(gts, preds, thr):
    """Enumerate every one-to-one matching; best = (count, total IoU)."""
    best = (0, 0.0)
    ng, npred = len(gts), len(preds)
    ious = [[box_iou(g, p) for p in preds] for g in gts]
    for r in range(min(ng, npred) + 1):
        for gsel in itertools.combinations(range(ng), r):
            for psel in itertools.permutations(range(npred), r):
                if all(ious[g][p] >= thr - 1e-9 for g, p in zip(gsel, psel)):
                    cand = (r, sum(ious[g][p] for g, p in zip(gsel, psel)))
                    if cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1] + 1e-12):
                        best = cand
    return best


class MatchingTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(match_boxes([], [], 0.5)[0], [])
        self.assertEqual(match_boxes([B(0, 0, 0.2, 0.2)], [], 0.5)[0], [])
        self.assertEqual(match_boxes([], [B(0, 0, 0.2, 0.2)], 0.5)[0], [])

    def test_iou_exactly_at_threshold_counts(self):
        # prediction covers exactly half of the GT area -> IoU == 0.5 (dyadic, exact)
        g = B(0.0, 0.0, 0.5, 0.5)
        p = B(0.0, 0.0, 0.25, 0.5)
        self.assertEqual(box_iou(g, p), 0.5)
        m, _ = match_boxes([g], [p], 0.5)
        self.assertEqual(len(m), 1)

    def test_iou_just_below_threshold_rejected(self):
        g = B(0.0, 0.0, 0.5, 0.5)
        p = B(0.0, 0.0, 0.2499, 0.5)  # IoU = 0.4998
        self.assertLess(box_iou(g, p), 0.5)
        self.assertEqual(match_boxes([g], [p], 0.5)[0], [])
        self.assertEqual(len(match_boxes([g], [p], 0.49)[0]), 1)

    def test_one_to_one_duplicate_predictions(self):
        # two predictions on one GT -> 1 TP, 1 unmatched prediction (FP candidate)
        g = [B(0.1, 0.1, 0.3, 0.3)]
        p = [B(0.1, 0.1, 0.3, 0.3), B(0.11, 0.1, 0.31, 0.3)]
        m, _ = match_boxes(g, p, 0.5)
        self.assertEqual([(a, b) for a, b, _ in m], [(0, 0)])

    def test_max_cardinality_beats_greedy_iou(self):
        # Greedy by highest IoU would pair g0-p0 (IoU 0.9) and leave g1 unmatched.
        # Optimal: g0-p1 and g1-p0 -> 2 matches.
        g0 = B(0.10, 0.10, 0.30, 0.30)
        g1 = B(0.14, 0.10, 0.34, 0.30)
        p0 = B(0.11, 0.10, 0.31, 0.30)
        p1 = B(0.05, 0.10, 0.25, 0.30)
        thr = 0.55
        ious = [[box_iou(g, p) for p in (p0, p1)] for g in (g0, g1)]
        self.assertGreater(ious[0][0], ious[0][1])  # greedy would take g0-p0 first
        self.assertTrue(ious[0][1] >= thr and ious[1][0] >= thr and ious[1][1] < thr)
        m, _ = match_boxes([g0, g1], [p0, p1], thr)
        self.assertEqual([(a, b) for a, b, _ in m], [(0, 1), (1, 0)])

    def test_secondary_objective_total_iou(self):
        # Both assignments give 2 matches; choose the one with larger IoU sum.
        g = [B(0.0, 0.0, 0.2, 0.2), B(0.5, 0.5, 0.7, 0.7)]
        p = [B(0.5, 0.5, 0.7, 0.69), B(0.0, 0.0, 0.2, 0.19)]
        m, _ = match_boxes(g, p, 0.1)
        self.assertEqual([(a, b) for a, b, _ in m], [(0, 1), (1, 0)])

    def test_multi_gt_multi_pred_with_fn_and_fp_coexisting(self):
        g = [B(0.0, 0.0, 0.2, 0.2), B(0.6, 0.6, 0.8, 0.8)]
        p = [B(0.0, 0.0, 0.2, 0.2), B(0.3, 0.3, 0.4, 0.4)]
        m, _ = match_boxes(g, p, 0.5)
        self.assertEqual([(a, b) for a, b, _ in m], [(0, 0)])  # TP=1, FN=1 (g1), FP=1 (p1)

    def test_more_gt_than_predictions(self):
        g = [B(0.0, 0.0, 0.2, 0.2), B(0.01, 0.0, 0.21, 0.2), B(0.5, 0.5, 0.6, 0.6)]
        p = [B(0.005, 0.0, 0.205, 0.2)]
        m, _ = match_boxes(g, p, 0.5)
        self.assertEqual(len(m), 1)

    def test_deterministic(self):
        rng = random.Random(1)
        for _ in range(50):
            g = [self._rand_box(rng) for _ in range(rng.randint(0, 5))]
            p = [self._rand_box(rng) for _ in range(rng.randint(0, 5))]
            self.assertEqual(match_boxes(g, p, 0.3)[0], match_boxes(g, p, 0.3)[0])

    @staticmethod
    def _rand_box(rng):
        x, y = rng.uniform(0, 0.6), rng.uniform(0, 0.6)
        return B(x, y, x + rng.uniform(0.05, 0.3), y + rng.uniform(0.05, 0.3))

    def test_random_against_brute_force(self):
        rng = random.Random(12345)
        for trial in range(400):
            thr = rng.choice([0.1, 0.3, 0.5, 0.7])
            # clustered boxes so that many pairs compete
            g = [self._rand_box(rng) for _ in range(rng.randint(0, 5))]
            p = [self._rand_box(rng) for _ in range(rng.randint(0, 5))]
            m, _ = match_boxes(g, p, thr)
            self.assertEqual(len({a for a, _, _ in m}), len(m))
            self.assertEqual(len({b for _, b, _ in m}), len(m))
            self.assertTrue(all(i >= thr - 1e-9 for _, _, i in m))
            cnt, tot = brute_force(g, p, thr)
            self.assertEqual(len(m), cnt, "trial %d cardinality" % trial)
            self.assertAlmostEqual(sum(i for _, _, i in m), tot, places=9)


class LabelFormatTests(unittest.TestCase):
    def test_roundtrip_b_style(self):
        text = "0 0.49855280 0.39157607 0.11663058 0.13720899\n0 0.05963110 0.48309323 0.11926221 0.24618646\n"
        self.assertEqual(format_boxes(parse_label_text(text)).decode(), text)

    def test_empty_file(self):
        self.assertEqual(format_boxes([]), b"")
        self.assertEqual(parse_label_text(""), [])

    def test_validation(self):
        with self.assertRaises(LabelError):
            validate_box({"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0, "h": 0.1})
        with self.assertRaises(LabelError):
            validate_box({"class_id": 0, "cx": 0.95, "cy": 0.5, "w": 0.2, "h": 0.1})
        with self.assertRaises(LabelError):
            validate_box({"class_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1})
        with self.assertRaises(LabelError):
            validate_box({"class_id": 0, "cx": float("nan"), "cy": 0.5, "w": 0.1, "h": 0.1})
        # legacy B box whose left edge is -5e-9 after 8-decimal rounding stays valid
        validate_box({"class_id": 0, "cx": 0.05963110, "cy": 0.48309323, "w": 0.11926221, "h": 0.24618646})


if __name__ == "__main__":
    unittest.main()
