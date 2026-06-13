# ============================================================
# Faster R-CNN — Test Split: Confusion Matrix + PR Eğrisi
#
# Bu script Google Colab'da (GPU runtime) çalıştırılmak üzere tasarlanmıştır.
#
# NEDEN BU SCRIPT VAR?
#   `train_fasterrcnn_raw_baseline.py` zaten test_metrics.csv (P/R/mAP50/
#   mAP50-95/F1) ve per_class_metrics.png üretiyor. AMA YOLOv8 (Ultralytics)
#   `model.val()` çağrısında confusion_matrix_normalized.png ve
#   BoxPR_curve.png'yi OTOMATİK üretiyor — bu, Ultralytics'in dahili plot
#   fonksiyonlarından geliyor ve torchvision modellerinde karşılığı yok.
#   Bu script, aynı iki görseli (confusion matrix + PR eğrisi) Faster R-CNN
#   için ELLE üretir, böylece "ortak değerlendirme" (adım 5) iki model için
#   de aynı görsel setine sahip olur.
#
# NE YAPAR?
#   1) Eğitilmiş best.pt ağırlığını yükler (yeniden eğitim YOK — sadece
#      inference).
#   2) Test split üzerinde TÜM tahminleri (eşiksiz, ham skorlarla) tek
#      geçişte toplar.
#   3) confidence>=0.5, IoU>=0.5 eşiğinde confusion matrix (background dahil,
#      3x3: background/Mass/Suspicious Calcification) üretir — satır bazlı
#      normalize (YOLOv8 confusion_matrix_normalized.png ile aynı yorumlama:
#      "gerçek sınıfın X'i model tarafından Y olarak tahmin edildi").
#   4) confidence eşiğini 0.05-0.95 arasında tarayarak sınıf bazlı PR eğrisi
#      üretir (YOLOv8 BoxPR_curve.png ile karşılaştırılabilir formatta).
#
# ÇIKTI: runs/<RUN_NAME>/test_eval/confusion_matrix.png, pr_curve.png
#   (per_class_metrics.png ile aynı klasörde, train scriptinin ürettiği
#   diğer test_eval çıktılarının yanında.)
#
# YENİDEN KULLANIM (crop+CLAHE ablasyonu, adım 6):
#   Aşağıdaki RUN_NAME / DATASET_VARIANT değişkenlerini güncelleyip aynı
#   scripti tekrar çalıştırmak yeterli (best.pt o run'ın train/weights/
#   klasöründe olmalı).
# ============================================================

from google.colab import drive
drive.mount("/content/drive")


from pathlib import Path
import random
import shutil

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.utils.data
import torchvision
from torchvision.transforms import functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from pycocotools.coco import COCO

# ------------------------------------------------------------
# 1. Yollar — train scriptiyle aynı kurallar.
#    crop+CLAHE ablasyonu için RUN_NAME ve DATASET_VARIANT'ı değiştirin.
# ------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = Path("/content/drive/MyDrive/vindr_mammo")
DATASET_BASE = PROJECT_ROOT / "dataset"
PREPARED_BASE = DATASET_BASE / "prepared_datasets"

SUBSET_NAME = "target2class_subset_v2_medium_balanced"
DATASET_VARIANT = "raw_png_2class"

RUN_NAME = "fasterrcnn_raw_baseline"

DRIVE_DATASET_DIR = PREPARED_BASE / f"{SUBSET_NAME}_{DATASET_VARIANT}"
LOCAL_DATASET_DIR = Path(f"/content/vindr_prepared_datasets/{SUBSET_NAME}_{DATASET_VARIANT}")

RUNS_DIR = PROJECT_ROOT / "runs"
RUN_DIR = RUNS_DIR / RUN_NAME
TEST_EVAL_DIR = RUN_DIR / "test_eval"
WEIGHTS_DIR = RUN_DIR / "train" / "weights"
TEST_EVAL_DIR.mkdir(parents=True, exist_ok=True)

print("best.pt:", WEIGHTS_DIR / "best.pt", "exists:", (WEIGHTS_DIR / "best.pt").exists())

if LOCAL_DATASET_DIR.exists():
    print("Local dataset zaten var, kopyalama atlanıyor:", LOCAL_DATASET_DIR)
else:
    print("Drive -> local kopyalanıyor...")
    shutil.copytree(DRIVE_DATASET_DIR, LOCAL_DATASET_DIR)
    print("Kopyalama tamamlandı:", LOCAL_DATASET_DIR)

# COCO category_id: 1=Mass, 2=Suspicious Calcification (0=background)
CLASS_NAMES = {1: "Mass", 2: "Suspicious Calcification"}
NUM_CLASSES = len(CLASS_NAMES) + 1

# ------------------------------------------------------------
# 2. Dataset — train scriptindeki CocoDetectionDataset ile aynı,
#    augmentasyon yok (test split).
# ------------------------------------------------------------


class CocoDetectionDataset(torch.utils.data.Dataset):
    """annotations/instances_<split>.json (COCO formatı) + images/<split>/*.png
    okur, {boxes: [x1,y1,x2,y2] (piksel), labels: int64} formatına çevirir.
    Detaylar için train_fasterrcnn_raw_baseline.py'deki aynı sınıfa bakınız.
    """

    def __init__(self, dataset_dir, split):
        self.dataset_dir = Path(dataset_dir)
        ann_file = self.dataset_dir / "annotations" / f"instances_{split}.json"
        self.coco = COCO(str(ann_file))
        self.image_ids = sorted(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = self.dataset_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            boxes.append([x, y, x + bw, y + bh])
            labels.append(int(ann["category_id"]))

        image = TF.to_tensor(image)

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes_t, "labels": labels_t, "image_id": torch.tensor([img_id])}
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


test_ds = CocoDetectionDataset(LOCAL_DATASET_DIR, "test")
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn,
)
print(f"test={len(test_ds)}")

# ------------------------------------------------------------
# 3. Modeli yükle (yalnızca inference — eğitim YOK)
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model.load_state_dict(torch.load(WEIGHTS_DIR / "best.pt", map_location=device))
model.to(device)
model.eval()

# ------------------------------------------------------------
# 4. Test seti üzerinde TEK geçişte tüm tahminleri topla (eşiksiz)
# ------------------------------------------------------------

all_preds, all_targets = [], []

with torch.no_grad():
    for images, targets in test_loader:
        images_dev = [img.to(device) for img in images]
        outputs = model(images_dev)
        for out, tgt in zip(outputs, targets):
            all_preds.append({
                "boxes": out["boxes"].cpu(),
                "scores": out["scores"].cpu(),
                "labels": out["labels"].cpu(),
            })
            all_targets.append({"boxes": tgt["boxes"], "labels": tgt["labels"]})

print(f"Test görüntü sayısı: {len(all_preds)}")

# ------------------------------------------------------------
# 5. Greedy eşleştirme yardımcı fonksiyonu
#
# Confidence'a göre sıralı tahminleri, IoU>=iou_threshold ile en iyi
# eşleşen (henüz eşleşmemiş) GT kutusuna atar. Çıktı: her biri
# (gt_label, pred_label) çifti — 0 = background:
#   (c, c)   -> doğru tespit (TP, sınıf c)
#   (c, 0)   -> kaçırılan GT (FN, sınıf c)
#   (0, c)   -> yanlış pozitif (FP, sınıf c)
#   (c1, c2) -> sınıf karışıklığı (c1 != c2): c1 için FN, c2 için FP
#
# NOT: Bu, sınıflar arası rekabeti de hesaba katan "global" bir eşleştirme
# (Ultralytics/COCO'nun kullandığı tam algoritmadan farklı, ama aynı
# mantık ailesinden — confidence-sıralı greedy IoU eşleştirmesi).
# ------------------------------------------------------------


def match_image(pred, tgt, conf_threshold, iou_threshold=0.5):
    scores = pred["scores"]
    keep = scores >= conf_threshold
    p_boxes = pred["boxes"][keep]
    p_labels = pred["labels"][keep]
    p_scores = scores[keep]

    g_boxes = tgt["boxes"]
    g_labels = tgt["labels"]

    order = torch.argsort(p_scores, descending=True)
    matched_gt = set()
    pairs = []

    for pi in order.tolist():
        if len(g_boxes) == 0:
            pairs.append((0, int(p_labels[pi])))
            continue
        ious = box_iou(p_boxes[pi:pi + 1], g_boxes)[0].clone()
        for gi in matched_gt:
            ious[gi] = -1
        best_iou, best_gi = ious.max(0)
        best_gi = int(best_gi)
        if float(best_iou) >= iou_threshold:
            pairs.append((int(g_labels[best_gi]), int(p_labels[pi])))
            matched_gt.add(best_gi)
        else:
            pairs.append((0, int(p_labels[pi])))

    for gi in range(len(g_boxes)):
        if gi not in matched_gt:
            pairs.append((int(g_labels[gi]), 0))

    return pairs


# ------------------------------------------------------------
# 6. Confusion Matrix (conf>=0.5, IoU>=0.5), satır bazlı normalize
# ------------------------------------------------------------

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

cm_label_ids = [0, 1, 2]
cm_label_names = ["background"] + [CLASS_NAMES[c] for c in sorted(CLASS_NAMES)]
cm = np.zeros((3, 3), dtype=int)  # [gerçek][tahmin]

for pred, tgt in zip(all_preds, all_targets):
    for gt_l, pred_l in match_image(pred, tgt, CONF_THRESHOLD, IOU_THRESHOLD):
        cm[cm_label_ids.index(gt_l)][cm_label_ids.index(pred_l)] += 1

row_sums = cm.sum(axis=1, keepdims=True)
cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(3))
ax.set_xticklabels(cm_label_names, rotation=30, ha="right")
ax.set_yticks(range(3))
ax.set_yticklabels(cm_label_names)
ax.set_xlabel("Tahmin")
ax.set_ylabel("Gerçek")
ax.set_title(f"Faster R-CNN — Confusion Matrix (normalize)\nconf>={CONF_THRESHOLD}, IoU>={IOU_THRESHOLD}")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cm_norm[i, j]:.2f}\n(n={cm[i, j]})", ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(TEST_EVAL_DIR / "confusion_matrix.png", dpi=130, bbox_inches="tight")
plt.close()
print("Kaydedildi:", TEST_EVAL_DIR / "confusion_matrix.png")

# ------------------------------------------------------------
# 7. PR Eğrisi — confidence eşiğini 0.05-0.95 arasında tarayarak
#    sınıf bazlı precision/recall hesapla
# ------------------------------------------------------------

thresholds = np.linspace(0.05, 0.95, 19)
pr_points = {c: {"precision": [], "recall": []} for c in CLASS_NAMES}

for t in thresholds:
    tp = {c: 0 for c in CLASS_NAMES}
    fp = {c: 0 for c in CLASS_NAMES}
    fn = {c: 0 for c in CLASS_NAMES}
    for pred, tgt in zip(all_preds, all_targets):
        for gt_l, pred_l in match_image(pred, tgt, float(t), IOU_THRESHOLD):
            if gt_l == pred_l and gt_l != 0:
                tp[gt_l] += 1
            else:
                if gt_l != 0:
                    fn[gt_l] += 1
                if pred_l != 0:
                    fp[pred_l] += 1
    for c in CLASS_NAMES:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 1.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        pr_points[c]["precision"].append(p)
        pr_points[c]["recall"].append(r)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
for c, name in CLASS_NAMES.items():
    r = np.array(pr_points[c]["recall"])
    p = np.array(pr_points[c]["precision"])
    order = np.argsort(r)
    ax.plot(r[order], p[order], marker="o", label=name)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title(f"Faster R-CNN — PR Eğrisi (Test, IoU>={IOU_THRESHOLD})")
ax.legend()
plt.tight_layout()
plt.savefig(TEST_EVAL_DIR / "pr_curve.png", dpi=130, bbox_inches="tight")
plt.close()
print("Kaydedildi:", TEST_EVAL_DIR / "pr_curve.png")

print("\nTamamlandı:", TEST_EVAL_DIR)
print("  ├── per_class_metrics.png   (train scriptinden)")
print("  ├── confusion_matrix.png    (bu script)")
print("  └── pr_curve.png            (bu script)")
