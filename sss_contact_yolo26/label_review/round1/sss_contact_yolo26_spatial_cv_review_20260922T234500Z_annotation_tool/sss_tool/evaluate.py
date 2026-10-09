"""Per-patch TP / FP-candidate / FN / TN computation from saved OOF predictions
and the *current* A labels. Predictions are never modified.

Definitions (at confidence threshold c and IoU threshold t):
  predictions considered = merged OOF boxes with confidence >= c
  TP  = matched GT/prediction pair (one-to-one, see matching.py)
  FN  = unmatched current-A GT box
  FP candidate = unmatched prediction; "false positive relative to the current
        annotation set", NOT a confirmed physical false positive
  TN  = scored patch with 0 GT and 0 predictions
  unscored patches (no OOF evaluation) never enter TP/FP/FN/TN.
"""
import collections
import csv
import io
import threading
import zipfile

from .common import ZIP_ROOT
from .matching import match_boxes

UNLABELLED = "unlabelled_model_candidate"
EXTRA = "extra_prediction_on_labelled_patch"


def evaluate_patch(entry, gt_boxes, conf, iou_thr):
    res = {"scored": entry["scored"], "gt_count": len(gt_boxes)}
    if not entry["scored"]:
        res.update({"pred_ks": [], "pred_count": 0, "tp": 0, "fp": 0, "fn": 0, "tn": False,
                    "matches": [], "unmatched_gt": [], "unmatched_pred": [], "fp_type": None,
                    "max_fp_conf": None, "max_pred_conf": None, "min_tp_iou": None, "best_iou_gt": []})
        return res
    preds = [p for p in entry["predictions"] if p["confidence"] >= conf]
    matches, ious = match_boxes(gt_boxes, preds, iou_thr)
    matched_g = {g for g, _, _ in matches}
    matched_p = {p for _, p, _ in matches}
    unmatched_pred = [preds[i]["k"] for i in range(len(preds)) if i not in matched_p]
    fp_type = None
    if unmatched_pred:
        fp_type = UNLABELLED if not gt_boxes else EXTRA
    res.update({
        "pred_ks": [p["k"] for p in preds],
        "pred_count": len(preds),
        "tp": len(matches),
        "fp": len(unmatched_pred),
        "fn": len(gt_boxes) - len(matches),
        "tn": not gt_boxes and not preds,
        "matches": [{"gt": g, "k": preds[p]["k"], "iou": v} for g, p, v in matches],
        "unmatched_gt": [g for g in range(len(gt_boxes)) if g not in matched_g],
        "unmatched_pred": unmatched_pred,
        "fp_type": fp_type,
        "max_fp_conf": max((entry["predictions"][k]["confidence"] for k in unmatched_pred), default=None),
        "max_pred_conf": preds[0]["confidence"] if preds else None,
        "min_tp_iou": min((v for _, _, v in matches), default=None),
        # best IoU of each GT against any considered prediction (explains FN)
        "best_iou_gt": [max(r) if r else 0.0 for r in ious],
    })
    return res


class Evaluator:
    def __init__(self, index, store, max_cache=6):
        self.index = index
        self.store = store
        self.cache = collections.OrderedDict()
        self.max_cache = max_cache
        self.lock = threading.Lock()
        store.listeners.append(self.invalidate)

    @staticmethod
    def key(conf, iou):
        return (round(float(conf), 4), round(float(iou), 4))

    def invalidate(self, pid):
        with self.lock:
            for res in self.cache.values():
                res.pop(pid, None)

    def results(self, conf, iou):
        k = self.key(conf, iou)
        with self.lock:
            res = self.cache.get(k)
            if res is None:
                res = {}
                self.cache[k] = res
                while len(self.cache) > self.max_cache:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(k)
        patches = self.index["patches"]
        if len(res) != len(patches):
            for pid, e in patches.items():
                if pid not in res:
                    res[pid] = evaluate_patch(e, self.store.a_boxes[pid], k[0], k[1])
        return res

    def patch(self, pid, conf, iou):
        return self.results(conf, iou)[pid]


def summarize(pids, results, index):
    s = collections.Counter()
    for pid in pids:
        r = results[pid]
        s["patches"] += 1
        s["gt_boxes"] += r["gt_count"]
        if not r["scored"]:
            s["unscored_patches"] += 1
            s["unscored_gt_boxes"] += r["gt_count"]
            continue
        s["scored_patches"] += 1
        s["pred_boxes"] += r["pred_count"]
        s["tp_boxes"] += r["tp"]
        s["fp_boxes"] += r["fp"]
        s["fn_boxes"] += r["fn"]
        s["tp_patches"] += r["tp"] > 0
        s["fp_patches"] += r["fp"] > 0
        s["fn_patches"] += r["fn"] > 0
        s["tn_patches"] += r["tn"]
        if r["fp"]:
            if r["fp_type"] == UNLABELLED:
                s["fp_unlabelled_boxes"] += r["fp"]
                s["fp_unlabelled_patches"] += 1
            else:
                s["fp_extra_boxes"] += r["fp"]
                s["fp_extra_patches"] += 1
        if r["fp"] and r["fn"]:
            s["fp_and_fn_patches"] += 1
    return dict(s)


def compare_with_zip_baseline(index, zip_path, b_boxes, conf=0.20, iou=0.50):
    """Recompute the baseline from B labels and compare with every ZIP artefact."""
    patches = index["patches"]
    res = {pid: evaluate_patch(e, b_boxes[pid], conf, iou) for pid, e in patches.items()}
    ours = summarize(list(patches), res, index)
    zman = index["zip_review_manifest"]["counts_on_deduplicated_oof_union_at_review_threshold"]
    totals = {
        "TP": {"ours": {"patch_count": ours.get("tp_patches", 0), "box_count": ours.get("tp_boxes", 0)}, "zip": zman["TP"]},
        "FP": {"ours": {"patch_count": ours.get("fp_patches", 0), "box_count": ours.get("fp_boxes", 0)}, "zip": zman["FP"]},
        "FN": {"ours": {"patch_count": ours.get("fn_patches", 0), "box_count": ours.get("fn_boxes", 0)}, "zip": zman["FN"]},
        "TN": {"ours": {"patch_count": ours.get("tn_patches", 0), "box_count": 0}, "zip": zman["TN"]},
    }
    for v in totals.values():
        v["equal"] = v["ours"] == v["zip"]

    z = zipfile.ZipFile(zip_path)
    rd = lambda n: list(csv.DictReader(io.StringIO(z.read(ZIP_ROOT + n).decode())))
    fp_csv = {(r["patch_id"], int(r["prediction_index"])): r for r in rd("html_review/fp_candidates.csv")}
    fn_csv = {(r["patch_id"], int(r["gt_index"])) for r in rd("html_review/fn.csv")}
    tp_csv = {(r["patch_id"], int(r["gt_index"]), int(r["prediction_index"])): float(r["iou"]) for r in rd("html_review/tp.csv")}
    tn_csv = {r["patch_id"] for r in rd("html_review/tn.csv")}
    uns_csv = {r["patch_id"] for r in rd("html_review/unscored.csv")}
    z.close()

    our_fp = {(p, k): r["fp_type"] for p, r in res.items() for k in r["unmatched_pred"]}
    our_fn = {(p, g) for p, r in res.items() for g in r["unmatched_gt"]}
    our_tp = {(p, m["gt"], m["k"]): m["iou"] for p, r in res.items() for m in r["matches"]}
    our_tn = {p for p, r in res.items() if r["tn"]}
    our_uns = {p for p, r in res.items() if not r["scored"]}

    fp_type_diff = [(p, k, our_fp[(p, k)], fp_csv[(p, k)]["candidate_type"]) for (p, k) in set(our_fp) & set(fp_csv)
                    if our_fp[(p, k)] != fp_csv[(p, k)]["candidate_type"]]
    fp_conf_diff = [(p, k) for (p, k) in set(our_fp) & set(fp_csv)
                    if abs(patches[p]["predictions"][k]["confidence"] - float(fp_csv[(p, k)]["prediction_confidence"])) > 1e-6]
    tp_iou_diff = [(key, our_tp[key], tp_csv[key]) for key in set(our_tp) & set(tp_csv) if abs(our_tp[key] - tp_csv[key]) > 1e-6]

    per_patch = []
    for pid, e in patches.items():
        if not e["scored"]:
            continue
        zb = e["zip_baseline"]
        r = res[pid]
        zm = sorted((m["gt_index"], m["prediction_index"]) for m in zb["matches"])
        om = sorted((m["gt"], m["k"]) for m in r["matches"])
        if (zb["tp_count"], zb["fp_count"], zb["fn_count"], zb["gt_count"], zb["prediction_count"]) != \
                (r["tp"], r["fp"], r["fn"], r["gt_count"], r["pred_count"]) or zm != om:
            per_patch.append({"patch_id": pid, "zip": {"tp": zb["tp_count"], "fp": zb["fp_count"], "fn": zb["fn_count"],
                                                        "matches": zm},
                              "ours": {"tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "matches": om}})

    def setdiff(a, b):
        return {"only_platform": sorted(map(list, a - b)) if a and isinstance(next(iter(a)), tuple) else sorted(a - b),
                "only_zip": sorted(map(list, b - a)) if b and isinstance(next(iter(b)), tuple) else sorted(b - a)}

    out = {
        "thresholds": {"confidence": conf, "iou": iou},
        "labels_used": "frozen backup B (identical to the ZIP ground_truth_labels)",
        "matching": "platform: optimal one-to-one (max #matches, then max total IoU)",
        "totals": totals,
        "platform_summary": ours,
        "fp_candidates_csv": {"rows": len(fp_csv), "platform_boxes": len(our_fp), **setdiff(set(our_fp), set(fp_csv)),
                              "candidate_type_differences": fp_type_diff, "confidence_differences": fp_conf_diff},
        "fn_csv": {"rows": len(fn_csv), "platform_boxes": len(our_fn), **setdiff(our_fn, fn_csv)},
        "tp_csv": {"rows": len(tp_csv), "platform_pairs": len(our_tp), **setdiff(set(our_tp), set(tp_csv)),
                   "iou_differences": tp_iou_diff},
        "tn_csv": {"rows": len(tn_csv), "platform_patches": len(our_tn), **setdiff(our_tn, tn_csv)},
        "unscored_csv": {"rows": len(uns_csv), "platform_patches": len(our_uns), **setdiff(our_uns, uns_csv)},
        "review_data_baseline_per_patch_differences": per_patch,
    }
    out["all_equal"] = (all(v["equal"] for v in totals.values()) and not per_patch and not fp_type_diff
                        and not fp_conf_diff and not tp_iou_diff
                        and all(not out[k]["only_platform"] and not out[k]["only_zip"]
                                for k in ("fp_candidates_csv", "fn_csv", "tp_csv", "tn_csv", "unscored_csv")))
    return out
