# val（chunks 8–9）误报 patch：红虚线 = 误报（未与任何 GT 一对一匹配），绿 = 正确检出，黄 = GT
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.optimize import linear_sum_assignment

CONF, IOU_T = 0.25, 0.50       # 显示/判定用的置信度阈值；F1 最优约 0.555，可改
N_SHOW, START = 30, 0          # 显示 30 个 patch；START=30 看下一批
img_dir, lab_dir = BUNDLE_ROOT / 'data' / 'images' / 'val', BUNDLE_ROOT / 'data' / 'labels' / 'val'
best_model.model.ch = 1
if hasattr(best_model.model, 'yaml'):
    best_model.model.yaml['channels'] = 1

def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)

rows = []
for r in best_model.predict([str(p) for p in sorted(img_dir.glob('*.png'))], imgsz=TRAIN_IMG_SIZE, conf=CONF,
                            iou=0.7, device=DEVICE, stream=True, verbose=False):
    lab = lab_dir / (Path(r.path).stem + '.txt')
    gts = [[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] for _, cx, cy, w, h in
           (map(float, l.split()) for l in lab.read_text().splitlines() if l.strip())]
    preds = list(zip(r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy()))
    matched = set()
    if gts and preds:   # 一对一匹配：先最大化匹配数，再最大化总 IoU
        m = np.array([[iou(g, b) for b, _ in preds] for g in gts])
        wgt = np.where(m >= IOU_T, min(m.shape) + 1 + m, 0)
        matched = {int(p) for g, p in zip(*linear_sum_assignment(wgt, maximize=True)) if wgt[g, p] > 0}
    fps = [c for k, (_, c) in enumerate(preds) if k not in matched]
    if fps:
        rows.append(dict(path=r.path, gts=gts, preds=preds, matched=matched, n_fp=len(fps), max_fp=max(fps)))

rows.sort(key=lambda d: -d['max_fp'])       # 最有把握的误报排在前面（最可能是漏标）
print(f'conf ≥ {CONF}: {sum(d["n_fp"] for d in rows)} 个误报框，分布在 {len(rows)} 个 val patch；'
      f'其中 GT 为空的 patch {sum(not d["gts"] for d in rows)} 个')

def rect(ax, b, color, lw, ls='-'):
    ax.add_patch(Rectangle((b[0] * 150, b[1] * 150), (b[2] - b[0]) * 150, (b[3] - b[1]) * 150,
                           fill=False, color=color, lw=lw, ls=ls))

fig, axes = plt.subplots(5, 6, figsize=(19, 16.5), constrained_layout=True)
for ax in axes.ravel():
    ax.axis('off')
for ax, d in zip(axes.ravel(), rows[START:START + N_SHOW]):
    ax.imshow(Image.open(d['path']), cmap='gray', vmin=0, vmax=255)
    for g in d['gts']:
        rect(ax, g, 'yellow', 1.5)
    for k, (b, c) in enumerate(d['preds']):
        ok = k in d['matched']
        rect(ax, b, 'lime' if ok else 'red', 1.8, '-' if ok else '--')
        ax.text(b[0] * 150, b[1] * 150 - 2, f'{c:.2f}', color='lime' if ok else 'red', fontsize=9, weight='bold')
    ax.set_title(f'{Path(d["path"]).stem[10:]}  GT {len(d["gts"])}  FP {d["n_fp"]}', fontsize=9)
plt.show()
