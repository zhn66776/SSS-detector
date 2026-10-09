#!/usr/bin/env python3
"""Generate sss_contact_yolo26_spatial_cv_A_v002.ipynb (round-2 spatial CV on chunks 1-9 with A_v002 labels)."""
import json
import os

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md(r"""
# Round-2 spatial CV for SSS label review with YOLO26n

- Input: `MyDrive/sss_contact_yolo26_spatial_split_A_v002.zip`, using the first-round reviewed labels for chunks 1–9.
- Eight adjacent two-chunk validation windows (1–2 through 8–9); train each fold on the other seven chunks, recomputing a 15 m spatial buffer.
- Test chunks 10–11 contribute geometry only for buffering. Their images and labels are not read.
- Settings match the first review round: single-channel YOLO26n, imgsz 256, AdamW lr0 1e-4, 80 maximum epochs, patience 20, 2× positive-patch oversampling and the same augmentation.
- Save each fold's out-of-fold predictions and best checkpoint to Drive immediately. Completed folds can be resumed without retraining.
- Download the final review ZIP and retain a Drive copy. Its layout is compatible with the local review platform.
- An unmatched prediction is a possible missing label or model error. This is label auditing, not independent performance evaluation.
""")

code(r"""
# 1. Install the same package versions as the first review round
%pip -q install ultralytics==8.4.159 pandas==2.2.3 pyyaml scipy pillow tqdm
""")

code(r"""
# 2. Mount Drive, extract data, read chunks 1-9 and configure training
from google.colab import drive
from pathlib import Path
from zipfile import ZipFile
from datetime import datetime, timezone
from collections import defaultdict, Counter
import json, math, random, shutil, zipfile
import numpy as np, pandas as pd, torch, yaml
from PIL import Image
from tqdm.auto import tqdm

drive.mount('/content/drive')
ZIP_PATH = Path('/content/drive/MyDrive/sss_contact_yolo26_spatial_split_A_v002.zip')
DRIVE_OUT = Path('/content/drive/MyDrive/sss_contact_yolo26_cv_A_v002')      # Per-fold results and final review ZIP
EXTRACT_ROOT = Path('/content/sss_cv_a_v002')
if not (EXTRACT_ROOT / 'sss_contact_yolo26_spatial_split_A_v002').is_dir():
    with ZipFile(ZIP_PATH) as zf:
        assert zf.testzip() is None, 'ZIP integrity check failed'
        zf.extractall(EXTRACT_ROOT)
BUNDLE_ROOT = EXTRACT_ROOT / 'sss_contact_yolo26_spatial_split_A_v002'
LOCAL_RUNS = Path('/content/sss_cv_a_v002_runs')
DRIVE_OUT.mkdir(parents=True, exist_ok=True)

MODEL_WEIGHTS, TRAIN_IMG_SIZE, SEED, POSITIVE_REPEAT = 'yolo26n.pt', 256, 42, 2
CUDA = torch.cuda.is_available()
DEVICE, BATCH_SIZE, WORKERS = (0, 256, 2) if CUDA else ('cpu', 8, 0)
TRAIN_ARGS = dict(epochs=80, imgsz=TRAIN_IMG_SIZE, batch=BATCH_SIZE, device=DEVICE, workers=WORKERS,
                  optimizer='AdamW', lr0=1e-4, lrf=1e-2, weight_decay=1e-3, warmup_epochs=3.0, cos_lr=True,
                  patience=20, seed=SEED, deterministic=True, pretrained=True, amp=CUDA, cache=False, plots=True,
                  close_mosaic=10, rect=False, degrees=10.0, translate=0.05, scale=0.20, shear=0.0,
                  perspective=0.0, fliplr=0.50, flipud=0.25, mosaic=0.25, mixup=0.0, copy_paste=0.0,
                  hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
BUFFER_M, PRED_CONF_FLOOR, NMS_IOU = 15.0, 0.05, 0.50
REVIEW_CONF, MATCH_IOU = 0.20, 0.50
DEV_CHUNKS, TEST_CHUNKS = list(range(1, 10)), [10, 11]
CV_FOLDS = [dict(fold_id=f'fold_{i:02d}_val_{i}_{i+1}', validation_chunks=[i, i + 1],
                 train_chunks=[c for c in DEV_CHUNKS if c not in (i, i + 1)]) for i in range(1, 9)]

manifest = pd.read_csv(BUNDLE_ROOT / 'metadata' / 'split_manifest_A_v002.csv')
dev = manifest[manifest.spatial_chunk.isin(DEV_CHUNKS)].copy()
test_geom = manifest[manifest.spatial_chunk.isin(TEST_CHUNKS)][['patch_id', 'x0', 'x1', 'y0', 'y1']]   # Geometry only
image_by_id = {r.patch_id: BUNDLE_ROOT / 'data' / 'images' / r.final_split / f'{r.patch_id}.png' for r in dev.itertuples()}
label_by_id = {r.patch_id: BUNDLE_ROOT / 'data' / 'labels' / r.final_split / f'{r.patch_id}.txt' for r in dev.itertuples()}
assert len(dev) == 9000 and all(p.is_file() for p in image_by_id.values()) and all(p.is_file() for p in label_by_id.values())
group_by_chunk = dict(manifest.groupby('spatial_chunk').original_group.first())
sample = Image.open(next(iter(image_by_id.values())))
assert sample.mode == 'L' and sample.size == (150, 150)
print('GPU:', torch.cuda.get_device_name(0) if CUDA else 'CPU', '| chunks 1–9 patches:', len(dev),
      '| test geometry rows (buffering only):', len(test_geom))
""")

code(r"""
# 3. Recompute each fold's 15 m buffers between train/validation and against test geometry
from scipy.spatial import cKDTree

def near_masks(a, b, buf=BUFFER_M):
    ca = np.c_[(a.x0 + a.x1) / 2, (a.y0 + a.y1) / 2]; ha = np.c_[(a.x1 - a.x0) / 2, (a.y1 - a.y0) / 2]
    cb = np.c_[(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2]; hb = np.c_[(b.x1 - b.x0) / 2, (b.y1 - b.y0) / 2]
    radius = math.hypot(buf + ha[:, 0].max() + hb[:, 0].max(), buf + ha[:, 1].max() + hb[:, 1].max())
    near_a, near_b = np.zeros(len(a), bool), np.zeros(len(b), bool)
    for i, js in enumerate(cKDTree(cb).query_ball_point(ca, r=radius)):
        for j in js:
            gap = np.maximum(np.abs(ca[i] - cb[j]) - ha[i] - hb[j], 0)
            if math.hypot(*gap) <= buf:
                near_a[i] = near_b[j] = True
    return near_a, near_b

FOLD_PLANS, plan_rows = [], []
for spec in CV_FOLDS:
    tr = dev[dev.spatial_chunk.isin(spec['train_chunks'])].reset_index(drop=True)
    va = dev[dev.spatial_chunk.isin(spec['validation_chunks'])].reset_index(drop=True)
    tr_near_va, va_near_tr = near_masks(tr, va)
    tr_keep = ~(tr_near_va | near_masks(tr, test_geom)[0])
    va_keep = ~(va_near_tr | near_masks(va, test_geom)[0])
    plan = dict(spec, train_ids=sorted(tr.patch_id[tr_keep]), val_ids=sorted(va.patch_id[va_keep]))
    FOLD_PLANS.append(plan)
    plan_rows.append(dict(fold_id=spec['fold_id'], train_chunks=','.join(map(str, spec['train_chunks'])),
                          validation_chunks=','.join(map(str, spec['validation_chunks'])),
                          train_patches_after_buffer=int(tr_keep.sum()), validation_patches_after_buffer=int(va_keep.sum()),
                          train_buffer_removed=int((~tr_keep).sum()), validation_buffer_removed=int((~va_keep).sum())))
fold_plan = pd.DataFrame(plan_rows)
display(fold_plan)
""")

code(r"""
# 4. Train folds and save OOF predictions to Drive; skip completed folds
from ultralytics import YOLO

def has_box(pid):
    return any(l.strip() for l in label_by_id[pid].read_text().splitlines())

FOLD_PREDICTIONS, FOLD_SUMMARIES = {}, []
for plan in FOLD_PLANS:
    fid = plan['fold_id']
    out = DRIVE_OUT / 'folds' / fid
    if not (out / 'fold_manifest.json').is_file():
        work = LOCAL_RUNS / 'inputs' / fid
        shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
        pos = [p for p in plan['train_ids'] if has_box(p)]
        pos_set = set(pos)
        entries = pos * POSITIVE_REPEAT + [p for p in plan['train_ids'] if p not in pos_set]
        random.Random(SEED).shuffle(entries)
        (work / 'train.txt').write_text(''.join(f'{image_by_id[p]}\n' for p in entries))
        (work / 'val.txt').write_text(''.join(f'{image_by_id[p]}\n' for p in plan['val_ids']))
        (work / 'data.yaml').write_text(yaml.safe_dump(dict(path=str(BUNDLE_ROOT), train=str(work / 'train.txt'),
                                        val=str(work / 'val.txt'), nc=1, names=['contact'], channels=1)))
        model = YOLO(MODEL_WEIGHTS, task='detect')          # Start every fold from pretrained weights
        model.train(data=str(work / 'data.yaml'), project=str(LOCAL_RUNS / 'train'), name=fid, exist_ok=True, **TRAIN_ARGS)
        best = YOLO(str(model.trainer.best), task='detect')
        best.model.ch = 1
        if hasattr(best.model, 'yaml'):
            best.model.yaml['channels'] = 1
        rows = []
        for pid, r in zip(plan['val_ids'], best.predict([str(image_by_id[p]) for p in plan['val_ids']], imgsz=TRAIN_IMG_SIZE,
                          conf=PRED_CONF_FLOOR, iou=NMS_IOU, device=DEVICE, batch=128, stream=True, verbose=False)):
            for k, (xywh, c, s) in enumerate(zip(r.boxes.xywhn.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())):
                rows.append(dict(fold_id=fid, patch_id=pid, class_id=int(c), cx=float(xywh[0]), cy=float(xywh[1]),
                                 w=float(xywh[2]), h=float(xywh[3]), confidence=float(s), prediction_index=k))
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=['fold_id', 'patch_id', 'class_id', 'cx', 'cy', 'w', 'h', 'confidence',
                                    'prediction_index']).to_csv(out / 'predictions.csv', index=False)
        shutil.copy2(model.trainer.best, out / 'best.pt')
        shutil.copy2(Path(model.trainer.save_dir) / 'results.csv', out / 'results.csv')
        metrics = {k: float(v) for k, v in getattr(model.trainer.metrics, 'results_dict', {}).items()}
        (out / 'fold_manifest.json').write_text(json.dumps(dict(
            fold_id=fid, train_patches_after_buffer=len(plan['train_ids']), validation_patches_after_buffer=len(plan['val_ids']),
            unique_positive_train_patches=len(pos), oversampled_train_entries=len(entries), metrics=metrics), indent=1))
        print('finished', fid)
    else:
        print('skip (already on Drive):', fid)
    summary = json.loads((out / 'fold_manifest.json').read_text())
    preds = pd.read_csv(out / 'predictions.csv')
    by_pid = {pid: [] for pid in plan['val_ids']}
    for r in preds.itertuples():
        by_pid[r.patch_id].append(dict(class_id=int(r.class_id), cx=r.cx, cy=r.cy, w=r.w, h=r.h,
                                       confidence=r.confidence, prediction_index=int(r.prediction_index)))
    FOLD_PREDICTIONS[fid] = by_pid
    FOLD_SUMMARIES.append(dict(plan_rows[FOLD_PLANS.index(plan)], **{k: summary[k] for k in
                          ('unique_positive_train_patches', 'oversampled_train_entries')}, **summary['metrics']))
print('Completed', len(FOLD_PREDICTIONS), 'folds')
""")

code(r"""
# 5. Merge OOF predictions and match to A_v002 labels (conf 0.20 / IoU 0.50)
from scipy.optimize import linear_sum_assignment

def box_iou(a, b):
    iw = min(a['cx'] + a['w'] / 2, b['cx'] + b['w'] / 2) - max(a['cx'] - a['w'] / 2, b['cx'] - b['w'] / 2)
    ih = min(a['cy'] + a['h'] / 2, b['cy'] + b['h'] / 2) - max(a['cy'] - a['h'] / 2, b['cy'] - b['h'] / 2)
    inter = max(iw, 0) * max(ih, 0)
    return inter / (a['w'] * a['h'] + b['w'] * b['h'] - inter) if inter > 0 else 0.0

def evaluate(gt, all_preds, conf=REVIEW_CONF, thr=MATCH_IOU):
    preds = [p for p in all_preds if p['confidence'] >= conf]
    iou = np.array([[box_iou(g, p) for p in preds] for g in gt]).reshape(len(gt), len(preds))
    matches = {}
    if iou.size:   # One-to-one: maximize match count, then total IoU
        weight = np.where(iou >= thr - 1e-9, min(iou.shape) + 1 + iou, 0.0)
        for g, p in zip(*linear_sum_assignment(weight, maximize=True)):
            if weight[g, p] > 0:
                matches[int(g)] = (int(p), float(iou[g, p]))
    un_gt = [g for g in range(len(gt)) if g not in matches]
    un_p = [p for p in range(len(preds)) if p not in {m[0] for m in matches.values()}]
    return dict(predictions=preds, matches=matches, unmatched_gt=un_gt, unmatched_predictions=un_p,
                best_iou={g: float(iou[g].max()) if len(preds) else 0.0 for g in un_gt},
                tp=len(matches), fp=len(un_p), fn=len(un_gt))

def merge_folds(runs, thr=MATCH_IOU):   # Merge overlapping cross-fold predictions and record fold support
    cands = sorted([dict(p, fold_id=fid) for fid, ps in runs for p in ps],
                   key=lambda p: (-p['confidence'], p['fold_id'], p['prediction_index']))
    clusters = []
    for c in cands:
        opts = [(box_iou(cl['rep'], c), i) for i, cl in enumerate(clusters)]
        opts = [o for o in opts if o[0] >= thr]
        if opts:
            clusters[max(opts)[1]]['members'].append(c)
        else:
            clusters.append(dict(rep=c, members=[c]))
    return [dict(cl['rep'], fold_ids=sorted({m['fold_id'] for m in cl['members']}),
                 fold_support=len({m['fold_id'] for m in cl['members']}), source_prediction_count=len(cl['members']))
            for cl in clusters]

def read_gt(pid):
    return [dict(class_id=int(v[0]), cx=v[1], cy=v[2], w=v[3], h=v[4]) for v in
            (list(map(float, l.split())) for l in label_by_id[pid].read_text().splitlines() if l.strip())]

info = dev.set_index('patch_id')
gt_by_id = {pid: read_gt(pid) for pid in dev.patch_id}
runs_by_pid = defaultdict(list)
pred_rows, eval_rows = [], []
for fid, by_pid in FOLD_PREDICTIONS.items():
    for pid, preds in by_pid.items():
        runs_by_pid[pid].append((fid, preds))
        common = dict(patch_id=pid, spatial_chunk=int(info.at[pid, 'spatial_chunk']),
                      annotation_group=info.at[pid, 'original_group'], fold_id=fid)
        e = evaluate(gt_by_id[pid], preds)
        eval_rows.append(dict(common, gt_count=len(gt_by_id[pid]), prediction_count=len(e['predictions']),
                              tp_count=e['tp'], fp_count=e['fp'], fn_count=e['fn']))
        pred_rows += [dict(common, **p) for p in preds]

scored = sorted(runs_by_pid)
unscored = sorted(set(dev.patch_id) - set(scored))
records, agree_rows, fn_rows, fp_rows, tp_rows, tn_rows = [], [], [], [], [], []
counts = Counter()
for pid in scored:
    gt, runs = gt_by_id[pid], sorted(runs_by_pid[pid])
    union = merge_folds(runs)
    res = evaluate(gt, union)
    per_fold = [evaluate(gt, ps) for _, ps in runs]
    ctype = ('unlabelled_model_candidate' if not gt else 'extra_prediction_on_labelled_patch') if res['fp'] else ''
    base = dict(patch_id=pid, spatial_chunk=int(info.at[pid, 'spatial_chunk']), annotation_group=info.at[pid, 'original_group'])
    unmatched_conf = [f['predictions'][p]['confidence'] for f in per_fold for p in f['unmatched_predictions']]
    agree_rows.append(dict(base, folds_evaluated=len(runs), folds_with_fp=sum(f['fp'] > 0 for f in per_fold),
                           folds_with_fn=sum(f['fn'] > 0 for f in per_fold), folds_with_tp=sum(f['tp'] > 0 for f in per_fold),
                           max_unmatched_confidence=max(unmatched_conf, default=''), candidate_type=ctype))
    is_tn = not gt and not res['predictions']
    for name, boxes in (('tp', res['tp']), ('fp', res['fp']), ('fn', res['fn'])):
        counts[f'{name}_boxes'] += boxes; counts[f'{name}_patches'] += boxes > 0
    counts['tn_patches'] += is_tn
    common = dict(base, cv_fold_ids=','.join(f for f, _ in runs), cv_fold_count=len(runs), gt_count=len(gt),
                  prediction_count=len(res['predictions']), tp_count=res['tp'], fp_count=res['fp'], fn_count=res['fn'],
                  confidence_threshold=REVIEW_CONF, iou_threshold=MATCH_IOU)
    fn_rows += [dict(common, gt_index=g, best_iou=res['best_iou'][g]) for g in res['unmatched_gt']]
    fp_rows += [dict(common, candidate_type=ctype, review_status='unreviewed', prediction_index=p,
                     prediction_confidence=res['predictions'][p]['confidence'],
                     prediction_fold_support=res['predictions'][p]['fold_support'],
                     prediction_fold_ids=','.join(res['predictions'][p]['fold_ids'])) for p in res['unmatched_predictions']]
    tp_rows += [dict(common, gt_index=g, prediction_index=p, iou=v, prediction_confidence=res['predictions'][p]['confidence'],
                     prediction_fold_support=res['predictions'][p]['fold_support']) for g, (p, v) in sorted(res['matches'].items())]
    if is_tn:
        tn_rows.append(dict(common, review_status='not_applicable'))
    records.append(dict(
        patch_id=pid, image_url=f'../validation/images/{pid}.png', image_path=f'validation/images/{pid}.png',
        ground_truth_label_path=f'validation/ground_truth_labels/{pid}.txt',
        prediction_label_path=f'validation/predicted_labels/{pid}.txt', width=150, height=150, spatial_x=None, spatial_y=None,
        spatial_chunk=base['spatial_chunk'], annotation_group=base['annotation_group'], cv_fold_ids=[f for f, _ in runs],
        cv_fold_count=len(runs), cv_summary={k: agree_rows[-1][k] for k in ('folds_with_fp', 'folds_with_fn', 'folds_with_tp')},
        ground_truth=gt, predictions=union,
        baseline=dict(gt_count=len(gt), prediction_count=len(res['predictions']), tp_count=res['tp'], fp_count=res['fp'],
                      fn_count=res['fn'], unmatched_gt_indices=res['unmatched_gt'],
                      unmatched_prediction_indices=res['unmatched_predictions'],
                      matches=[dict(gt_index=g, prediction_index=p, iou=v) for g, (p, v) in sorted(res['matches'].items())],
                      best_iou_for_unmatched_gt={str(g): v for g, v in res['best_iou'].items()}, candidate_type=ctype or None,
                      best_unmatched_prediction_confidence=max((res['predictions'][p]['confidence']
                                                                for p in res['unmatched_predictions']), default=None))))
print('OOF scored:', len(scored), '/ 9000 | unscored:', len(unscored))
print({k: counts[k] for k in ('tp_boxes', 'tp_patches', 'fp_boxes', 'fp_patches', 'fn_boxes', 'fn_patches', 'tn_patches')})
""")

code(r"""
# 6. Package the review ZIP in the round-1 layout and download it
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
PKG = LOCAL_RUNS / 'package' / 'sss_contact_yolo26_spatial_cv_review'   # Preserve the root directory name expected by the review platform
shutil.rmtree(PKG.parent, ignore_errors=True)
for sub in ['validation/images', 'validation/ground_truth_labels', 'validation/predicted_labels', 'html_review']:
    (PKG / sub).mkdir(parents=True)
union_by_id = {r['patch_id']: r['predictions'] for r in records}
for pid in tqdm(sorted(dev.patch_id), desc='package'):
    shutil.copy2(image_by_id[pid], PKG / 'validation' / 'images' / f'{pid}.png')
    shutil.copy2(label_by_id[pid], PKG / 'validation' / 'ground_truth_labels' / f'{pid}.txt')
    lines = [f"{p['class_id']} {p['cx']:.8f} {p['cy']:.8f} {p['w']:.8f} {p['h']:.8f} {p['confidence']:.8f}"
             for p in union_by_id.get(pid, [])]
    (PKG / 'validation' / 'predicted_labels' / f'{pid}.txt').write_text(''.join(l + '\n' for l in lines))

reason = 'not held out in any fold after fold-specific 15 m guards'
tables = {
    'development_split_manifest.csv': dev[['patch_id', 'original_group', 'spatial_chunk', 'final_split', 'x0', 'x1', 'y0', 'y1']],
    'fold_plan.csv': fold_plan, 'fold_metrics.csv': pd.DataFrame(FOLD_SUMMARIES),
    'predictions_by_fold.csv': pd.DataFrame(pred_rows)[['patch_id', 'spatial_chunk', 'annotation_group', 'fold_id', 'class_id',
                                                        'cx', 'cy', 'w', 'h', 'confidence', 'prediction_index']],
    'evaluation_by_fold.csv': pd.DataFrame(eval_rows), 'fold_agreement.csv': pd.DataFrame(agree_rows),
    'unscored.csv': pd.DataFrame(dict(patch_id=unscored, reason=reason)),
}
for name, df in tables.items():
    df.to_csv(PKG / name, index=False)
for name in ['fold_agreement.csv', 'fold_metrics.csv', 'unscored.csv']:
    shutil.copy2(PKG / name, PKG / 'html_review' / name)
cols = lambda extra: ['patch_id', 'spatial_chunk', 'annotation_group', 'cv_fold_ids', 'cv_fold_count'] + extra
pd.DataFrame(fn_rows, columns=cols(['gt_index', 'gt_count', 'prediction_count', 'tp_count', 'fp_count', 'fn_count', 'best_iou',
             'confidence_threshold', 'iou_threshold'])).to_csv(PKG / 'html_review' / 'fn.csv', index=False)
pd.DataFrame(fp_rows, columns=cols(['candidate_type', 'review_status', 'gt_count', 'prediction_count', 'tp_count', 'fp_count',
             'fn_count', 'prediction_index', 'prediction_confidence', 'prediction_fold_support', 'prediction_fold_ids',
             'confidence_threshold', 'iou_threshold'])).to_csv(PKG / 'html_review' / 'fp_candidates.csv', index=False)
pd.DataFrame(tp_rows, columns=cols(['gt_index', 'prediction_index', 'iou', 'prediction_confidence', 'prediction_fold_support',
             'gt_count', 'prediction_count', 'tp_count', 'fp_count', 'fn_count', 'confidence_threshold',
             'iou_threshold'])).to_csv(PKG / 'html_review' / 'tp.csv', index=False)
pd.DataFrame(tn_rows, columns=cols(['review_status', 'gt_count', 'prediction_count', 'tp_count', 'fp_count', 'fn_count',
             'confidence_threshold', 'iou_threshold'])).to_csv(PKG / 'html_review' / 'tn.csv', index=False)

(PKG / 'html_review' / 'review_data.json').write_text(json.dumps(dict(
    source_directory='downloaded CV review bundle', total_validation_images=len(dev), processed_validation_images=len(scored),
    baseline_confidence_threshold=REVIEW_CONF, baseline_iou_threshold=MATCH_IOU, prediction_confidence_floor=PRED_CONF_FLOOR,
    records=records, skipped=[dict(patch_id=p, reason=reason) for p in unscored]), separators=(',', ':')))
(PKG / 'html_review' / 'review_manifest.json').write_text(json.dumps(dict(
    tool='YOLO26n spatial CV OOF annotation audit (round 2, A_v002 labels)', source_drive_zip=ZIP_PATH.name,
    total_development_patches_chunks_1_9=len(dev), oof_scored_unique_patches=len(scored), unscored_patch_count=len(unscored),
    test_chunks=TEST_CHUNKS, test_images_labels_predictions_metrics_used=False,
    fold_design='8 adjacent two-chunk validation windows (1-2, 2-3, ..., 8-9); train the other 7 chunks',
    fold_prediction_appearances=len(eval_rows), image_size=dict(width=150, height=150, channels=1),
    training_image_size=TRAIN_IMG_SIZE, prediction_confidence_floor=PRED_CONF_FLOOR, review_confidence_threshold=REVIEW_CONF,
    nms_iou=NMS_IOU, matching_iou=MATCH_IOU, spatial_buffer_m=BUFFER_M,
    annotation_group_by_spatial_chunk={str(k): v for k, v in group_by_chunk.items()}, fold_metrics_csv='fold_metrics.csv',
    counts_on_deduplicated_oof_union_at_review_threshold={
        n.upper(): dict(patch_count=int(counts[f'{n}_patches']), box_count=int(counts.get(f'{n}_boxes', 0)))
        for n in ('fn', 'fp', 'tp', 'tn')},
    generation_time_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'), source_directory='downloaded CV review bundle',
    interpretation='OOF-union discovery output for manual label review; not an unbiased CV performance estimate.'), indent=2))
(PKG / 'html_review' / 'README.md').write_text('Round-2 YOLO26n spatial CV review bundle (labels = A_final_labels_v002).\n'
                                               'FP = unmatched relative to the current annotation set, not confirmed false positives.\n')

zip_path = LOCAL_RUNS / f'sss_contact_yolo26_spatial_cv_review_A_v002_{RUN_ID}.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
    for p in sorted(PKG.rglob('*')):
        if p.is_file():
            zf.write(p, p.relative_to(PKG.parent))
print('Review ZIP:', zip_path.name, f'({zip_path.stat().st_size / 2**20:.0f} MiB)')

SAVE_DRIVE_COPY = True          # Keep a Drive backup in case the browser download is interrupted
if SAVE_DRIVE_COPY:
    shutil.copy2(zip_path, DRIVE_OUT / zip_path.name)
    print('Drive backup:', DRIVE_OUT / zip_path.name)

from google.colab import files
files.download(str(zip_path))   # Download through the browser; large files may take some time
""")

code(r"""
# 7. Inspect the 30 highest-confidence patches with unmatched OOF predictions (conf >= 0.20)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from IPython.display import display

fp = pd.DataFrame(fp_rows)
print('Unmatched predictions (possible missing labels):', len(fp), 'boxes /', fp.patch_id.nunique(), 'patches')
display(fp.groupby('candidate_type').agg(boxes=('patch_id', 'size'), patches=('patch_id', 'nunique')))
display(pd.cut(fp.prediction_confidence, [0.2, 0.3, 0.5, 0.7, 1.0]).value_counts().sort_index().rename('boxes by confidence'))

top = fp.sort_values('prediction_confidence', ascending=False).drop_duplicates('patch_id').head(30)
rec_by_id = {r['patch_id']: r for r in records}
fig, axes = plt.subplots(5, 6, figsize=(19, 16.5), constrained_layout=True)
for ax in axes.ravel():
    ax.axis('off')
for ax, row in zip(axes.ravel(), top.itertuples()):
    rec = rec_by_id[row.patch_id]
    ax.imshow(Image.open(image_by_id[row.patch_id]), cmap='gray', vmin=0, vmax=255)
    for g in rec['ground_truth']:
        ax.add_patch(Rectangle(((g['cx'] - g['w'] / 2) * 150, (g['cy'] - g['h'] / 2) * 150), g['w'] * 150, g['h'] * 150,
                               fill=False, color='yellow', lw=1.5))
    unmatched = set(rec['baseline']['unmatched_prediction_indices'])
    for k, p in enumerate(p for p in rec['predictions'] if p['confidence'] >= REVIEW_CONF):
        color = 'red' if k in unmatched else 'lime'
        ax.add_patch(Rectangle(((p['cx'] - p['w'] / 2) * 150, (p['cy'] - p['h'] / 2) * 150), p['w'] * 150, p['h'] * 150,
                               fill=False, color=color, lw=1.8, ls='--' if k in unmatched else '-'))
        ax.text((p['cx'] - p['w'] / 2) * 150, (p['cy'] - p['h'] / 2) * 150 - 2, f"{p['confidence']:.2f}", color=color, fontsize=8)
    ax.set_title(f"{row.patch_id[10:]} chunk {row.spatial_chunk}", fontsize=9)
plt.show()
print('Yellow = A_v002 labels; red dashed = unmatched predictions; green = matched predictions')
""")

meta = {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"},
        "colab": {"provenance": [], "machine_shape": "hm", "gpuType": "L4"}, "accelerator": "GPU"}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sss_contact_yolo26_spatial_cv_A_v002.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump({"nbformat": 4, "nbformat_minor": 0, "metadata": meta, "cells": cells}, f, ensure_ascii=False, indent=1)
print(out)
