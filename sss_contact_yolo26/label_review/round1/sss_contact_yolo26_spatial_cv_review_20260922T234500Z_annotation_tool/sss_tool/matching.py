"""Deterministic one-to-one GT/prediction matching.

Objective (lexicographic):
  1. maximise the number of matched pairs, using only pairs with IoU >= threshold;
  2. among those, maximise the total IoU.

This is solved exactly as a maximum-weight assignment with weight
``K + IoU`` for eligible pairs (K = min(n_gt, n_pred) + 1) and 0 otherwise.
Because every extra match adds at least K > (sum of all IoUs), the optimum
always has maximum cardinality first. The Hungarian algorithm below is exact
and deterministic for a fixed input order (GT in file order, predictions in
descending confidence order), so exact ties are resolved identically every run.
"""

# Absorbs floating-point noise at the boundary (IoU exactly == threshold counts).
IOU_EPS = 1e-9


def box_iou(a, b):
    ax0, ax1 = a["cx"] - a["w"] / 2, a["cx"] + a["w"] / 2
    ay0, ay1 = a["cy"] - a["h"] / 2, a["cy"] + a["h"] / 2
    bx0, bx1 = b["cx"] - b["w"] / 2, b["cx"] + b["w"] / 2
    by0, by1 = b["cy"] - b["h"] / 2, b["cy"] + b["h"] / 2
    iw = min(ax1, bx1) - max(ax0, bx0)
    ih = min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _hungarian_max(weight):
    """Max-weight assignment for an n x m matrix (n <= m). Returns row -> col."""
    n = len(weight)
    m = len(weight[0]) if n else 0
    if n == 0:
        return []
    big = max(max(r) for r in weight)
    cost = [[big - w for w in row] for row in weight]  # minimise
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assign = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            assign[p[j] - 1] = j - 1
    return assign


def match_boxes(gts, preds, iou_threshold):
    """Return (matches, iou_matrix).

    matches: list of (gt_index, pred_index, iou) sorted by gt_index.
    iou_matrix[g][p]: IoU of every pair (for display of best IoU etc.).
    """
    ng, npred = len(gts), len(preds)
    ious = [[box_iou(g, p) for p in preds] for g in gts]
    if ng == 0 or npred == 0:
        return [], ious
    k = min(ng, npred) + 1.0
    eligible = [[ious[g][p] >= iou_threshold - IOU_EPS for p in range(npred)] for g in range(ng)]
    if not any(any(r) for r in eligible):
        return [], ious
    w = [[(k + ious[g][p]) if eligible[g][p] else 0.0 for p in range(npred)] for g in range(ng)]
    if ng <= npred:
        assign = _hungarian_max(w)
        pairs = [(g, assign[g]) for g in range(ng)]
    else:
        wt = [[w[g][p] for g in range(ng)] for p in range(npred)]
        assign = _hungarian_max(wt)
        pairs = [(assign[p], p) for p in range(npred)]
    matches = [(g, p, ious[g][p]) for g, p in pairs if p >= 0 and g >= 0 and eligible[g][p]]
    matches.sort()
    return matches, ious
