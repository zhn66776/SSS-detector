# val（chunks 8–9）漏报：黄粗框 = 漏检的 GT，黄细框 = 已检出的 GT，
# 绿 = 模型预测 ≥ CONF，灰虚线 = 模型低置信度预测 (0.05–CONF)，框上数字为置信度
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.optimize import linear_sum_assignment

CONF, IOU_T = 0.25, 0.50        # 与混淆矩阵相同的 0.25；Ultralytics 混淆矩阵的匹配规则略有不同，数量可能与 53 差几个
PER_PAGE = 30
img_dir, lab_dir = BUNDLE_ROOT / 'data' / 'images' / 'val', BUNDLE_ROOT / 'data' / 'labels' / 'val'
best_model.model.ch = 1
if hasattr(best_model.model, 'yaml'):
    best_model.model.yaml['channels'] = 1

def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)

labelled = [p for p in sorted(img_dir.glob('*.png')) if (lab_dir / f'{p.stem}.txt').read_text().strip()]
fn_rows = []
for r in best_model.predict([str(p) for p in labelled], imgsz=TRAIN_IMG_SIZE, conf=0.05, iou=0.7,
                            device=DEVICE, stream=True, verbose=False):
    lab = lab_dir / (Path(r.path).stem + '.txt')
    gts = [[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] for _, cx, cy, w, h in
           (map(float, l.split()) for l in lab.read_text().splitlines() if l.strip())]
    preds = [(b, c) for b, c in zip(r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    hi = [(b, c) for b, c in preds if c >= CONF]
    matched_gt = set()
    if hi:   # 一对一匹配（与混淆矩阵同一阈值）
        m = np.array([[iou(g, b) for b, _ in hi] for g in gts])
        wgt = np.where(m >= IOU_T, min(m.shape) + 1 + m, 0)
        matched_gt = {int(g) for g, p in zip(*linear_sum_assignment(wgt, maximize=True)) if wgt[g, p] > 0}
    for gi, g in enumerate(gts):
        if gi in matched_gt:
            continue
        overlaps = [(iou(g, b), c) for b, c in preds]
        best_conf = max((c for v, c in overlaps if v >= IOU_T), default=0.0)   # 模型在这个 GT 上的最高置信度
        best_iou_hi = max((v for (v, c) in overlaps if c >= CONF), default=0.0)
        fn_rows.append(dict(path=r.path, gts=gts, gi=gi, preds=preds, best_conf=best_conf, best_iou_hi=best_iou_hi))

kind = lambda d: ('low-conf' if d['best_conf'] > 0 else 'mislocalised' if d['best_iou_hi'] > 0.1 else 'not detected')
KIND_ZH = {'low-conf': '低置信度检出 (<CONF)', 'mislocalised': '框位置不对 (IoU<0.5)', 'not detected': '完全没检出'}
print(f'conf {CONF} / IoU {IOU_T}: {len(fn_rows)} 个漏检 GT，分布在 {len({d["path"] for d in fn_rows})} 个 patch')
for k in KIND_ZH:
    print(f'  {KIND_ZH[k]}: {sum(kind(d) == k for d in fn_rows)}')
fn_rows.sort(key=lambda d: (-d['best_conf'], -d['best_iou_hi']))

def rect(ax, b, color, lw, ls='-'):
    ax.add_patch(Rectangle((b[0] * 150, b[1] * 150), (b[2] - b[0]) * 150, (b[3] - b[1]) * 150,
                           fill=False, color=color, lw=lw, ls=ls))

for start in range(0, len(fn_rows), PER_PAGE):
    fig, axes = plt.subplots(5, 6, figsize=(19, 16.5), constrained_layout=True)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, d in zip(axes.ravel(), fn_rows[start:start + PER_PAGE]):
        ax.imshow(Image.open(d['path']), cmap='gray', vmin=0, vmax=255)
        for gi, g in enumerate(d['gts']):
            rect(ax, g, 'yellow', 3 if gi == d['gi'] else 1)
        for b, c in d['preds']:
            hi = c >= CONF
            rect(ax, b, 'lime' if hi else 'gray', 1.5, '-' if hi else '--')
            ax.text(b[0] * 150, b[1] * 150 - 2, f'{c:.2f}', color='lime' if hi else 'dimgray', fontsize=8, weight='bold')
        ax.set_title(f'{Path(d["path"]).stem[10:]} GT{d["gi"]}\n{kind(d)} (conf {d["best_conf"]:.2f}, IoU {d["best_iou_hi"]:.2f})',
                     fontsize=9, color='red')
    plt.show()
