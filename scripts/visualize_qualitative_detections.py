# ============================================================
# Qualitative Detection Görselleştirmesi — GT vs Tahmin Bbox Overlay
#
# Bu script Google Colab'da çalıştırılmak üzere tasarlanmıştır
# (GPU önerilir ama CPU'da da çalışır — sadece inference yapılıyor,
# eğitim YOK).
#
# NE YAPAR?
#   Literatürdeki "qualitative results" figürlerinin (Ribli et al. 2018
#   Fig.3-4, jimaging-11-00314 Fig.8-9, Abdikenov et al. 2025 Fig.6-7)
#   bir benzerini üretir: test split'ten seçilen örnek mamogramlar
#   üzerinde GERÇEK (GT, yeşil kesikli) kutular ile 3 eğitilmiş modelin
#   TAHMİNLERİNİ (kırmızı=Mass, turuncu=Suspicious Calcification,
#   confidence skorlu) yan yana karşılaştırır:
#     - YOLOv8s (raw_png_2class)
#     - Faster R-CNN ResNet50-FPN (raw_png_2class)
#     - Faster R-CNN ResNet50-FPN (crop_clahe_2class)
#
#   Crop+CLAHE modeli farklı (kırpılmış + CLAHE uygulanmış) bir görüntü
#   üzerinde çalıştığı için, aynı lezyon hem RAW hem CROP+CLAHE
#   versiyonunda (GT + tahmin) gösterilir — böylece crop+CLAHE'nin
#   görsel/niteliksel etkisi de doğrudan karşılaştırılabilir.
#
# NEDEN?
#   mAP/F1 gibi sayısal metrikler "ne kadar iyi" sorusunu cevaplıyor;
#   qualitative örnekler "nerede/nasıl hata yapıyor" (FP/FN/yanlış
#   sınıflandırma) sorusunu cevaplıyor ve sunumda görsel olarak en çok
#   dikkat çeken kısım — referans makalelerin hepsinde bu türde figürler
#   var.
#
# ÇIKTI: runs/comparison/qualitative_detections.png
#   (Colab'dan indirip projeyi yüklediğiniz klasördeki aynı yola koyun.)
# ============================================================

from google.colab import drive
drive.mount("/content/drive")


from pathlib import Path
import random
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import torch
import torchvision
from torchvision.transforms import functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from ultralytics import YOLO

# ------------------------------------------------------------
# 1. Yollar (diğer scriptlerle aynı kurallar)
# ------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path("/content/drive/MyDrive/vindr_mammo")
DATASET_BASE = PROJECT_ROOT / "dataset"
PREPARED_BASE = DATASET_BASE / "prepared_datasets"
SUBSET_NAME = "target2class_subset_v2_medium_balanced"

RAW_VARIANT = "raw_png_2class"
CROP_VARIANT = "crop_clahe_2class"

DRIVE_RAW_DIR = PREPARED_BASE / f"{SUBSET_NAME}_{RAW_VARIANT}"
DRIVE_CROP_DIR = PREPARED_BASE / f"{SUBSET_NAME}_{CROP_VARIANT}"
LOCAL_RAW_DIR = Path(f"/content/vindr_prepared_datasets/{SUBSET_NAME}_{RAW_VARIANT}")
LOCAL_CROP_DIR = Path(f"/content/vindr_prepared_datasets/{SUBSET_NAME}_{CROP_VARIANT}")

RUNS_DIR = PROJECT_ROOT / "runs"
OUT_DIR = RUNS_DIR / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHTS = RUNS_DIR / "yolov8s_raw_baseline" / "train" / "weights" / "best.pt"
FRCNN_RAW_WEIGHTS = RUNS_DIR / "fasterrcnn_raw_baseline" / "train" / "weights" / "best.pt"
FRCNN_CROP_WEIGHTS = RUNS_DIR / "fasterrcnn_crop_clahe" / "train" / "weights" / "best.pt"

for p in (YOLO_WEIGHTS, FRCNN_RAW_WEIGHTS, FRCNN_CROP_WEIGHTS):
    print(p, "exists:", p.exists())

# ------------------------------------------------------------
# 2. Veri setlerini Drive -> local kopyala (diğer scriptlerle aynı kural)
# ------------------------------------------------------------

for drive_dir, local_dir in [(DRIVE_RAW_DIR, LOCAL_RAW_DIR), (DRIVE_CROP_DIR, LOCAL_CROP_DIR)]:
    if local_dir.exists():
        print("Local dataset zaten var, kopyalama atlanıyor:", local_dir)
    else:
        print("Drive -> local kopyalanıyor:", local_dir)
        shutil.copytree(drive_dir, local_dir)

# COCO category_id: 1=Mass, 2=Suspicious Calcification (0=background)
CLASS_NAMES = {1: "Mass", 2: "Suspicious Calcification"}
# YOLOv8 (Ultralytics) sınıf id'leri 0-indeksli: 0=Mass, 1=Suspicious Calcification
YOLO_CLASS_NAMES = {0: "Mass", 1: "Suspicious Calcification"}
NUM_CLASSES = len(CLASS_NAMES) + 1  # Faster R-CNN için (+1 = background)

GT_COLOR = "#39FF14"  # neon yeşil — tüm GT kutuları için
PRED_COLORS = {"Mass": "#FF3B30", "Suspicious Calcification": "#FF9500"}

CONF_THRESHOLD_VIS = 0.3   # görselleştirme için (test_metrics.csv'deki 0.5'ten farklı,
                           # sadece "ne tür tahminler üretiliyor" göstermek için daha düşük)
MAX_BOXES_PER_PANEL = 6    # çok fazla kutu görseli kalabalıklaştırmasın

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------------------------------------------------
# 3. Modelleri yükle (yalnızca inference — eğitim YOK)
# ------------------------------------------------------------

yolo_model = YOLO(str(YOLO_WEIGHTS))


def load_frcnn(weights_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


frcnn_raw_model = load_frcnn(FRCNN_RAW_WEIGHTS)
frcnn_crop_model = load_frcnn(FRCNN_CROP_WEIGHTS)

# ------------------------------------------------------------
# 4. Örnek görüntü seçimi (test split, RAW ve CROP+CLAHE'de ortak image_id)
# ------------------------------------------------------------

raw_coco = COCO(str(LOCAL_RAW_DIR / "annotations" / "instances_test.json"))
crop_coco = COCO(str(LOCAL_CROP_DIR / "annotations" / "instances_test.json"))
crop_image_ids = set(crop_coco.imgs.keys())

mass_only, calc_only, both_classes = [], [], []
for img_id in raw_coco.imgs.keys():
    if img_id not in crop_image_ids:
        continue  # crop+CLAHE karşılığı yoksa (fallback) atla
    ann_ids = raw_coco.getAnnIds(imgIds=img_id)
    cats = {a["category_id"] for a in raw_coco.loadAnns(ann_ids)}
    if cats == {1}:
        mass_only.append(img_id)
    elif cats == {2}:
        calc_only.append(img_id)
    elif cats == {1, 2}:
        both_classes.append(img_id)

random.shuffle(mass_only)
random.shuffle(calc_only)
random.shuffle(both_classes)

N_PER_CATEGORY = 2
selected_ids = mass_only[:N_PER_CATEGORY] + calc_only[:N_PER_CATEGORY] + both_classes[:1]
print(f"Seçilen görüntü sayısı: {len(selected_ids)}  (mass_only={len(mass_only)}, "
      f"calc_only={len(calc_only)}, both={len(both_classes)} aday arasından)")

# ------------------------------------------------------------
# 5. Yardımcı fonksiyonlar: çizim + inference
# ------------------------------------------------------------


def draw_box(ax, box, label, color, score=None, linestyle="-"):
    x1, y1, x2, y2 = box
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              edgecolor=color, facecolor="none", linestyle=linestyle)
    ax.add_patch(rect)
    text = label if score is None else f"{label} {score:.2f}"
    ax.text(x1, max(0, y1 - 4), text, color=color, fontsize=7, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.55, pad=1, edgecolor="none"))


def draw_gt_coco(ax, coco, img_id):
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    for ann in anns:
        x, y, w, h = ann["bbox"]
        label = CLASS_NAMES[ann["category_id"]]
        draw_box(ax, (x, y, x + w, y + h), f"{label} (GT)", GT_COLOR, linestyle="--")


@torch.no_grad()
def frcnn_predict(model, img_path):
    image = Image.open(img_path).convert("RGB")
    tensor = TF.to_tensor(image).to(device)
    out = model([tensor])[0]
    scores = out["scores"].cpu().numpy()
    keep = scores >= CONF_THRESHOLD_VIS
    boxes = out["boxes"][keep].cpu().numpy()
    labels = out["labels"][keep].cpu().numpy()
    scores = scores[keep]
    order = np.argsort(-scores)[:MAX_BOXES_PER_PANEL]
    return [(boxes[i], CLASS_NAMES.get(int(labels[i]), str(labels[i])), float(scores[i])) for i in order]


def yolo_predict(img_path):
    result = yolo_model.predict(source=str(img_path), conf=CONF_THRESHOLD_VIS, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    order = np.argsort(-scores)[:MAX_BOXES_PER_PANEL]
    return [(boxes[i], YOLO_CLASS_NAMES.get(int(cls_ids[i]), str(cls_ids[i])), float(scores[i])) for i in order]


# ------------------------------------------------------------
# 6. Grid figür: her satır bir örnek görüntü, 5 sütun
#    [RAW+GT | YOLOv8s (raw) | FRCNN (raw) | CROP+CLAHE+GT | FRCNN (crop+CLAHE)]
# ------------------------------------------------------------

COL_TITLES = [
    "RAW — Ground Truth",
    "YOLOv8s (raw)",
    "Faster R-CNN (raw)",
    "Crop+CLAHE — Ground Truth",
    "Faster R-CNN (crop+CLAHE)",
]

n_rows = len(selected_ids)
fig, axes = plt.subplots(n_rows, 5, figsize=(24, 4.6 * n_rows))
if n_rows == 1:
    axes = axes.reshape(1, 5)

for row_idx, img_id in enumerate(selected_ids):
    raw_info = raw_coco.imgs[img_id]
    crop_info = crop_coco.imgs[img_id]
    raw_path = LOCAL_RAW_DIR / raw_info["file_name"]
    crop_path = LOCAL_CROP_DIR / crop_info["file_name"]

    raw_img = np.array(Image.open(raw_path).convert("L"))
    crop_img = np.array(Image.open(crop_path).convert("L"))

    cats_present = {a["category_id"] for a in raw_coco.loadAnns(raw_coco.getAnnIds(imgIds=img_id))}
    cat_label = "+".join(CLASS_NAMES[c] for c in sorted(cats_present))

    # Kolon 0: RAW + GT
    ax = axes[row_idx, 0]
    ax.imshow(raw_img, cmap="gray")
    draw_gt_coco(ax, raw_coco, img_id)
    ax.set_ylabel(f"img_id={img_id}\n[{cat_label}]", fontsize=9)

    # Kolon 1: YOLOv8s (raw) tahminleri
    ax = axes[row_idx, 1]
    ax.imshow(raw_img, cmap="gray")
    for box, label, score in yolo_predict(raw_path):
        draw_box(ax, box, label, PRED_COLORS.get(label, "white"), score=score)

    # Kolon 2: Faster R-CNN (raw) tahminleri
    ax = axes[row_idx, 2]
    ax.imshow(raw_img, cmap="gray")
    for box, label, score in frcnn_predict(frcnn_raw_model, raw_path):
        draw_box(ax, box, label, PRED_COLORS.get(label, "white"), score=score)

    # Kolon 3: CROP+CLAHE + GT
    ax = axes[row_idx, 3]
    ax.imshow(crop_img, cmap="gray")
    draw_gt_coco(ax, crop_coco, img_id)

    # Kolon 4: Faster R-CNN (crop+CLAHE) tahminleri
    ax = axes[row_idx, 4]
    ax.imshow(crop_img, cmap="gray")
    for box, label, score in frcnn_predict(frcnn_crop_model, crop_path):
        draw_box(ax, box, label, PRED_COLORS.get(label, "white"), score=score)

    for col_idx in range(5):
        axes[row_idx, col_idx].set_xticks([])
        axes[row_idx, col_idx].set_yticks([])
        if row_idx == 0:
            axes[row_idx, col_idx].set_title(COL_TITLES[col_idx], fontsize=10)

# Ortak lejant (figür üstü)
legend_handles = [
    patches.Patch(edgecolor=GT_COLOR, facecolor="none", linestyle="--", linewidth=2, label="Ground Truth"),
    patches.Patch(edgecolor=PRED_COLORS["Mass"], facecolor="none", linewidth=2, label="Tahmin: Mass"),
    patches.Patch(edgecolor=PRED_COLORS["Suspicious Calcification"], facecolor="none", linewidth=2,
                  label="Tahmin: Suspicious Calcification"),
]
fig.legend(handles=legend_handles, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.02))
fig.suptitle(f"Qualitative Detection Karşılaştırması (conf>={CONF_THRESHOLD_VIS}, test split)", y=1.05, fontsize=13)

plt.tight_layout()
out_path = OUT_DIR / "qualitative_detections.png"
plt.savefig(out_path, dpi=130, bbox_inches="tight")
plt.close()
print("\nKaydedildi:", out_path)
print("Bu dosyayı indirip projeyi yüklediğiniz klasördeki runs/comparison/ altına koyun.")
