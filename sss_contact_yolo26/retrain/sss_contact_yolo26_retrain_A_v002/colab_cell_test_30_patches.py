# test（spatial chunks 10–11）上抽 30 个 patch 看模型能否找到 contact —— 只出图，不做 evaluation
# 青色 = 模型预测（数字为置信度）；黄色虚线 = 现有 test 标签（尚未复核，仅供参考）
import random
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

CONF = 0.20            # 显示的最低置信度
N, SEED = 30, 0        # 改 SEED 换一批
SAMPLE_FROM = 'pred_or_gt'   # 'pred_or_gt' = 只从有预测或有标签的 patch 里抽；'all' = 全部 patch 随机抽
SHOW_GT = True

img_dir, lab_dir = BUNDLE_ROOT / 'data' / 'images' / 'test', BUNDLE_ROOT / 'data' / 'labels' / 'test'
best_model.model.ch = 1
if hasattr(best_model.model, 'yaml'):
    best_model.model.yaml['channels'] = 1

def read_gt(stem):
    rows = [list(map(float, l.split())) for l in (lab_dir / f'{stem}.txt').read_text().splitlines() if l.strip()]
    return [[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] for _, cx, cy, w, h in rows]

results = {}
for r in best_model.predict([str(p) for p in sorted(img_dir.glob('*.png'))], imgsz=TRAIN_IMG_SIZE, conf=CONF,
                            iou=0.7, device=DEVICE, stream=True, verbose=False):
    results[Path(r.path).stem] = (r.path, list(zip(r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy())))

pool = [s for s, (_, p) in results.items() if SAMPLE_FROM == 'all' or p or read_gt(s)]
picked = random.Random(SEED).sample(sorted(pool), min(N, len(pool)))

fig, axes = plt.subplots(5, 6, figsize=(19, 16.5), constrained_layout=True)
for ax in axes.ravel():
    ax.axis('off')
for ax, stem in zip(axes.ravel(), picked):
    path, preds = results[stem]
    ax.imshow(Image.open(path), cmap='gray', vmin=0, vmax=255)
    if SHOW_GT:
        for g in read_gt(stem):
            ax.add_patch(Rectangle((g[0] * 150, g[1] * 150), (g[2] - g[0]) * 150, (g[3] - g[1]) * 150,
                                   fill=False, color='yellow', lw=1.5, ls='--'))
    for b, c in preds:
        ax.add_patch(Rectangle((b[0] * 150, b[1] * 150), (b[2] - b[0]) * 150, (b[3] - b[1]) * 150,
                               fill=False, color='cyan', lw=1.8))
        ax.text(b[0] * 150, b[1] * 150 - 2, f'{c:.2f}', color='cyan', fontsize=9, weight='bold')
    ax.set_title(stem[10:], fontsize=9)
plt.show()
