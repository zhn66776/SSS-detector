# ===== Cell A: validation recall by label source (run training notebook cells 1-8 first) =====
import numpy as np
import pandas as pd
from pathlib import Path

SOURCES_CSV = Path('/content/drive/MyDrive/label_sources_A_v002.csv')   # Upload the CSV to MyDrive first
SPLIT = 'val'
CONF_T, IOU_T = 0.29, 0.50        # Confidence at the best validation F1 / matching IoU

gt = pd.read_csv(SOURCES_CSV).query('split == @SPLIT').copy()
gt['group'] = np.where(gt.source != 'adopted_prediction', gt.source,
                       np.where(gt.adopted_conf < 0.3, 'adopted_conf<0.3', 'adopted_conf>=0.3'))

best_model.model.ch = 1
if hasattr(best_model.model, 'yaml'):
    best_model.model.yaml['channels'] = 1
img_dir = BUNDLE_ROOT / 'data' / 'images' / SPLIT
preds = {}
for r in best_model.predict([str(img_dir / f'{p}.png') for p in sorted(gt.patch_id.unique())], imgsz=TRAIN_IMG_SIZE,
                            conf=0.05, iou=0.7, device=DEVICE, stream=True, verbose=False):
    preds[Path(r.path).stem] = (r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy())

def iou(box, others):
    x0, y0 = np.maximum(box[0], others[:, 0]), np.maximum(box[1], others[:, 1])
    x1, y1 = np.minimum(box[2], others[:, 2]), np.minimum(box[3], others[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(box) + area(others) - inter)

gt['xyxy'] = [np.array([r.cx - r.w / 2, r.cy - r.h / 2, r.cx + r.w / 2, r.cy + r.h / 2]) for r in gt.itertuples()]
model_conf, best_iou = [], []
for r in gt.itertuples():
    boxes, confs = preds[r.patch_id]
    ious = iou(r.xyxy, boxes) if len(boxes) else np.zeros(0)
    best_iou.append(float(ious.max()) if len(ious) else 0.0)
    model_conf.append(float(confs[ious >= IOU_T].max()) if (ious >= IOU_T).any() else 0.0)
gt['model_conf'], gt['best_iou'] = model_conf, best_iou
gt['detected'] = gt.model_conf >= CONF_T

summary = gt.groupby('group').agg(boxes=('detected', 'size'), detected=('detected', 'sum'))
summary['recall'] = (summary.detected / summary.boxes).round(3)
display(summary)
print(f'Overall recall @conf {CONF_T}: {gt.detected.mean():.3f} (each GT is assessed independently; not one-to-one matching; diagnostic only)')


# ===== Cell B: inspect labels adopted below confidence 0.3, starting with missed detections =====
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

look = gt[gt.group == 'adopted_conf<0.3'].sort_values(['detected', 'adopted_conf']).reset_index(drop=True)
look.drop(columns='xyxy').to_csv('/content/drive/MyDrive/low_conf_adopted_review_%s.csv' % SPLIT, index=False)
print(f'{len(look)} boxes; missed: {(~look.detected).sum()} boxes. Yellow = selected GT; thin orange = other GT in this patch; '
      f'cyan = new model predictions (>={CONF_T}); gray dashed = low-confidence predictions (0.05-{CONF_T})')

def rect(ax, b, color, lw, ls='-'):
    ax.add_patch(Rectangle((b[0] * 150, b[1] * 150), (b[2] - b[0]) * 150, (b[3] - b[1]) * 150,
                           fill=False, color=color, lw=lw, ls=ls))

PER_PAGE = 20
for start in range(0, len(look), PER_PAGE):
    fig, axes = plt.subplots(4, 5, figsize=(16, 13.5), constrained_layout=True)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, r in zip(axes.ravel(), look.iloc[start:start + PER_PAGE].itertuples()):
        ax.imshow(Image.open(img_dir / f'{r.patch_id}.png'), cmap='gray', vmin=0, vmax=255)
        for o in gt[gt.patch_id == r.patch_id].itertuples():
            if o.box_index != r.box_index:
                rect(ax, o.xyxy, 'orange', 1)
        boxes, confs = preds[r.patch_id]
        for b, c in zip(boxes, confs):
            rect(ax, b, 'cyan' if c >= CONF_T else 'gray', 1.2, '-' if c >= CONF_T else '--')
        rect(ax, r.xyxy, 'yellow', 2)
        ax.set_title(f'{r.patch_id[10:]} #{r.box_index}\nadopt {r.adopted_conf:.2f} | model {r.model_conf:.2f} '
                     f'{"found" if r.detected else "MISSED"}', fontsize=9, color='black' if r.detected else 'red')
    plt.show()
