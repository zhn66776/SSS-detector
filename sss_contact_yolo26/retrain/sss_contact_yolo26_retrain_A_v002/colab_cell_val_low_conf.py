# val 上新模型置信度 < 0.3 的预测：黄 = GT，红虚线 = 低置信度预测 (0.05–0.3)，青 = 预测 ≥ 0.3
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

LOW, HIGH, IOU_T = 0.05, 0.30, 0.50
img_dir, lab_dir = BUNDLE_ROOT / 'data' / 'images' / 'val', BUNDLE_ROOT / 'data' / 'labels' / 'val'
best_model.model.ch = 1
if hasattr(best_model.model, 'yaml'):
    best_model.model.yaml['channels'] = 1

def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)

rows = []
for r in best_model.predict([str(p) for p in sorted(img_dir.glob('*.png'))], imgsz=TRAIN_IMG_SIZE, conf=LOW,
                            iou=0.7, device=DEVICE, stream=True, verbose=False):
    lab = lab_dir / (Path(r.path).stem + '.txt')
    gts = [[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] for _, cx, cy, w, h in
           (map(float, l.split()) for l in lab.read_text().splitlines() if l.strip())]
    preds = list(zip(r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy()))
    low = [(b, c) for b, c in preds if c < HIGH]
    if low:
        rows.append(dict(path=r.path, gts=gts, preds=preds, max_low=max(c for _, c in low),
                         low_on_gt=sum(any(iou(b, g) >= IOU_T for g in gts) for b, _ in low), n_low=len(low)))

rows.sort(key=lambda d: -d['max_low'])
n_low = sum(d['n_low'] for d in rows); n_on_gt = sum(d['low_on_gt'] for d in rows)
print(f'{len(rows)} 个 val patch 含低置信度预测；共 {n_low} 个框，其中 {n_on_gt} 个与 GT 重叠 (IoU≥{IOU_T}) → '
      f'阈值降到 {LOW} 可多找回的 GT；其余 {n_low - n_on_gt} 个与任何 GT 都不重叠')

def rect(ax, b, color, lw, ls='-'):
    ax.add_patch(Rectangle((b[0] * 150, b[1] * 150), (b[2] - b[0]) * 150, (b[3] - b[1]) * 150,
                           fill=False, color=color, lw=lw, ls=ls))

MAX_SHOW, PER_PAGE = 60, 20                      # 最多显示 60 个 patch（按最高低置信度排序）
for start in range(0, min(len(rows), MAX_SHOW), PER_PAGE):
    fig, axes = plt.subplots(4, 5, figsize=(16, 13.5), constrained_layout=True)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, d in zip(axes.ravel(), rows[start:start + PER_PAGE]):
        ax.imshow(Image.open(d['path']), cmap='gray', vmin=0, vmax=255)
        for g in d['gts']:
            rect(ax, g, 'yellow', 2)
        for b, c in d['preds']:
            rect(ax, b, 'cyan' if c >= HIGH else 'red', 1.5, '-' if c >= HIGH else '--')
            ax.text(b[0] * 150, b[1] * 150 - 2, f'{c:.2f}', color='cyan' if c >= HIGH else 'red', fontsize=8)
        ax.set_title(f'{Path(d["path"]).stem[10:]}  GT {len(d["gts"])}  low {d["n_low"]} (on GT {d["low_on_gt"]})', fontsize=9)
    plt.show()
