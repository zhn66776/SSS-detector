#!/usr/bin/env python3
"""Generate the Colab notebook sss_contact_yolo26_retrain_A_round2_v001.ipynb."""
import json
import os

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md("""
# YOLO26n 单通道 SSS contact 重新训练（第二轮人工复核标签 A_round2_v001）

- 数据：`sss_contact_yolo26_spatial_split_A_round2_v001.zip`（上传到 Google Drive 的 `MyDrive/` 根目录）
- 拆分不变：spatial chunks 1–7 训练、8–9 validation、10–11 test；15 m buffer 不参与
- 标签：chunks 1–9 使用第二轮复核后的 `A_round2_labels_v001`；chunks 10–11（test）仍为原标签，本 notebook 不评估 test
- 模型配置与之前的 YOLO26n notebook 相同（imgsz 256、AdamW、lr0 1e-4、80 epochs、同样的增强、正样本 2× 过采样），便于和上一次比较
""")

code("""
# 1. 挂载 Drive，校验并解压数据 ZIP
from google.colab import drive
from pathlib import Path
from zipfile import ZipFile
import shutil

drive.mount('/content/drive')
ZIP_PATH = Path('/content/drive/MyDrive/sss_contact_yolo26_spatial_split_A_round2_v001.zip')
assert ZIP_PATH.exists(), f'Drive 中找不到 {ZIP_PATH}'

EXTRACT_ROOT = Path('/content/sss_a_round2_v001')
shutil.rmtree(EXTRACT_ROOT, ignore_errors=True)
with ZipFile(ZIP_PATH) as zf:
    assert zf.testzip() is None, 'ZIP 损坏'
    zf.extractall(EXTRACT_ROOT)

BUNDLE_ROOT = EXTRACT_ROOT / 'sss_contact_yolo26_spatial_split_A_round2_v001'
print('Bundle root:', BUNDLE_ROOT)
print((BUNDLE_ROOT / 'README.md').read_text())
""")

code("""
# 2. 安装支持 YOLO26 的 Ultralytics（如之前 import 过 ultralytics，请重启 runtime 后从第 1 个 cell 重跑）
%pip -q install -U 'ultralytics>=8.4.159,<8.5' 'pandas==2.2.3' matplotlib pyyaml
""")

code("""
# 3. 数据检查：各 split 数量 / 正样本 / 单通道 150×150
import pandas as pd
import torch
from PIL import Image
from IPython.display import display

def split_stats(split):
    labels = list((BUNDLE_ROOT / 'data' / 'labels' / split).glob('*.txt'))
    boxes = [sum(1 for l in p.read_text().splitlines() if l.strip()) for p in labels]
    images = list((BUNDLE_ROOT / 'data' / 'images' / split).glob('*.png'))
    return {'split': split, 'images': len(images), 'labels': len(labels),
            'positive_patches': sum(b > 0 for b in boxes), 'boxes': sum(boxes)}

display(pd.DataFrame([split_stats(s) for s in ['train', 'val', 'test', 'buffer']]))
sample = Image.open(next((BUNDLE_ROOT / 'data' / 'images' / 'train').glob('*.png')))
assert sample.mode == 'L' and sample.size == (150, 150), (sample.mode, sample.size)
print('Sample image:', sample.mode, sample.size)
print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
""")

code("""
# 4. 训练配置（与之前的 YOLO26n notebook 一致）
import json

DATA_YAML = BUNDLE_ROOT / 'dataset.yaml'
MODEL_WEIGHTS = 'yolo26n.pt'
RUN_ROOT = Path('/content/sss_contact_yolo26_runs')
TRAIN_IMG_SIZE = 256
SEED = 42
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = 0 if CUDA_AVAILABLE else 'cpu'
BATCH_SIZE = 256 if CUDA_AVAILABLE else 8

AUGMENTATION = dict(degrees=10.0, translate=0.05, scale=0.20, shear=0.0, perspective=0.0,
                    fliplr=0.50, flipud=0.25, mosaic=0.25, mixup=0.0, copy_paste=0.0,
                    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)

TRAIN_ARGS = dict(
    data=str(DATA_YAML), epochs=80, imgsz=TRAIN_IMG_SIZE, batch=BATCH_SIZE, device=DEVICE,
    workers=2 if CUDA_AVAILABLE else 0, optimizer='AdamW', lr0=1e-4, lrf=1e-2, weight_decay=1e-3,
    warmup_epochs=3.0, cos_lr=True, patience=20, seed=SEED, deterministic=True, pretrained=True,
    amp=CUDA_AVAILABLE, cache=False, plots=True, project=str(RUN_ROOT),
    name='yolo26n_A_round2_v001_pos2x', exist_ok=False, close_mosaic=10, rect=False, **AUGMENTATION,
)
print(json.dumps({k: TRAIN_ARGS[k] for k in ['epochs', 'imgsz', 'batch', 'optimizer', 'lr0', 'device']}, indent=1))
""")

code("""
# 5. 训练集正样本 2× 过采样（validation 不变）
import random

train_images = sorted((BUNDLE_ROOT / 'data' / 'images' / 'train').glob('*.png'))
def is_positive(img):
    lab = BUNDLE_ROOT / 'data' / 'labels' / 'train' / f'{img.stem}.txt'
    return any(l.strip() for l in lab.read_text().splitlines())

positives = [p for p in train_images if is_positive(p)]
negatives = [p for p in train_images if not is_positive(p)]
entries = positives * 2 + negatives
random.Random(SEED).shuffle(entries)

train_list = BUNDLE_ROOT / 'data' / 'train_oversampled_2x.txt'
train_list.write_text('\\n'.join(str(p) for p in entries) + '\\n')

spec = json.loads(DATA_YAML.read_text())
spec.update(path=str(BUNDLE_ROOT), train='data/train_oversampled_2x.txt')
DATA_YAML = BUNDLE_ROOT / 'dataset_oversampled_2x.yaml'
DATA_YAML.write_text(json.dumps(spec, indent=2))
TRAIN_ARGS['data'] = str(DATA_YAML)
print(f'positives {len(positives)}, negatives {len(negatives)}, training entries {len(entries)}')
""")

code("""
# 6. 加载 YOLO26n 预训练权重并训练
from ultralytics import YOLO

model = YOLO(MODEL_WEIGHTS, task='detect')
model.train(**TRAIN_ARGS)
RUN_DIR = Path(model.trainer.save_dir)
BEST_WEIGHTS = Path(model.trainer.best)
print('Run dir:', RUN_DIR)
print('Best weights:', BEST_WEIGHTS)
""")

code("""
# 7. Loss 曲线（train vs val）与 validation 指标随 epoch 变化
import matplotlib.pyplot as plt

hist = pd.read_csv(RUN_DIR / 'results.csv')
hist.columns = [c.strip() for c in hist.columns]
loss_names = sorted({c.split('/', 1)[1] for c in hist.columns if c.startswith('train/') and 'loss' in c})
metric_cols = [c for c in ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
               if c in hist.columns]

fig, axes = plt.subplots(1, len(loss_names) + 1, figsize=(5 * (len(loss_names) + 1), 4), constrained_layout=True)
for ax, name in zip(axes, loss_names):
    for split in ['train', 'val']:
        if f'{split}/{name}' in hist.columns:
            ax.plot(hist['epoch'], hist[f'{split}/{name}'], marker='o', ms=3, label=split)
    ax.set(title=name, xlabel='epoch'); ax.legend(); ax.grid(alpha=0.3)
for c in metric_cols:
    axes[-1].plot(hist['epoch'], hist[c], marker='o', ms=3, label=c.replace('metrics/', ''))
axes[-1].set(title='validation metrics (chunks 8-9)', xlabel='epoch'); axes[-1].legend(fontsize=8); axes[-1].grid(alpha=0.3)
fig.savefig(RUN_DIR / 'loss_and_val_metrics.png', dpi=150)
plt.show()
display(hist.tail(3))
""")

code("""
# 8. best.pt 在 validation（spatial chunks 8–9）上的最终评估
#    mAP / PR / F1 用 conf=0.001（标准做法）；混淆矩阵另用 CM_CONF 跑一次，否则会把 0.001 以上的所有预测都算成误报
from IPython.display import Image as DisplayImage

best_model = YOLO(str(BEST_WEIGHTS), task='detect')
val_args = dict(data=str(DATA_YAML), split='val', imgsz=TRAIN_IMG_SIZE, batch=BATCH_SIZE, device=DEVICE,
                iou=0.7, plots=True, project=str(RUN_ROOT), exist_ok=True)
val_metrics = best_model.val(conf=0.001, name='val_chunks8_9_A_round2', **val_args)
VAL_DIR = Path(val_metrics.save_dir)
display(pd.Series(val_metrics.results_dict, name='validation chunks 8-9').to_frame())
for pattern in ['*PR_curve.png', '*F1_curve.png']:
    for path in sorted(VAL_DIR.glob(pattern)):
        print(path.name); display(DisplayImage(filename=str(path), width=550))

CM_CONF = 0.25   # 可改成上面 F1 曲线最高点对应的置信度
cm_metrics = best_model.val(conf=CM_CONF, name=f'val_confusion_conf{CM_CONF}', **val_args)
print(f'confusion matrix @ conf {CM_CONF}（这次运行的 mAP 不要使用）')
display(DisplayImage(filename=str(Path(cm_metrics.save_dir) / 'confusion_matrix.png'), width=550))
""")

code("""
# 9. 把训练和 validation 结果保存到 Drive（Colab 本地文件断开后会丢失）
SAVE_ROOT = Path('/content/drive/MyDrive/sss_contact_yolo26_results_A_round2_v001')
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
for src in [RUN_DIR, VAL_DIR, Path(cm_metrics.save_dir)]:
    shutil.copytree(src, SAVE_ROOT / src.name, dirs_exist_ok=True)
print('Saved to', SAVE_ROOT, [p.name for p in SAVE_ROOT.iterdir()])
""")

meta = {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"},
        "colab": {"provenance": [], "machine_shape": "hm", "gpuType": "L4"}, "accelerator": "GPU"}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sss_contact_yolo26_retrain_A_round2_v001.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump({"nbformat": 4, "nbformat_minor": 0, "metadata": meta, "cells": cells}, f, ensure_ascii=False, indent=1)
print(out)
