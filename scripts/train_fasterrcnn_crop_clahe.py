# ============================================================
# Faster R-CNN (ResNet50-FPN) — Crop+CLAHE 2-Class Ablasyon (Mass + Suspicious Calcification)
#
# Bu script Google Colab'da (GPU runtime) çalıştırılmak üzere tasarlanmıştır.
#
# NEDEN BU SCRIPT VAR?
#   train_fasterrcnn_raw_baseline.py ile aynı iskelet — TEK FARK:
#   DATASET_VARIANT = "crop_clahe_2class" (göğüs bölgesine crop + CLAHE
#   kontrast iyileştirmesi uygulanmış veri seti).
#   Model karşılaştırmasında (runs/comparison) Faster R-CNN, YOLOv8'i tüm
#   metriklerde (özellikle Suspicious Calcification mAP50'de ~2.7x) geride
#   bıraktı -> "en iyi model" budur. Plan adımı 6 ("en iyi modeli crop+CLAHE
#   ile yeniden eğit") bu nedenle SADECE Faster R-CNN için uygulanıyor;
#   YOLOv8 tekrar eğitilmiyor.
#
# NOT: raw baseline'da test_metrics.csv + confusion matrix/PR eğrisi İKİ
# AYRI scriptle (train + evaluate_fasterrcnn.py) üretilmişti. Burada ayrı bir
# eval scripti çalıştırmamak için, confusion matrix + PR eğrisi üretimi
# (evaluate_fasterrcnn.py'deki match_image() mantığı) DOĞRUDAN bu scriptin
# sonuna eklendi — test split üzerindeki TEK inference geçişinden elde edilen
# ham tahminler (skorlu, eşiksiz) hem test_metrics.csv hem de
# confusion_matrix.png/pr_curve.png için kullanılıyor.
#
# NE YAPAR?
#   1) target2class_subset_v2_medium_balanced_crop_clahe_2class veri setini
#      Drive'dan Colab'in lokal diskine kopyalar (raw baseline scriptiyle
#      aynı kural — bu run da çok-epoch'lu bir eğitim olduğu için lokal
#      kopya I/O açısından faydalı).
#   2) Veri setiyle birlikte gelen hazır COCO formatlı etiketleri
#      (annotations/instances_{train,val,test}.json) pycocotools ile okuyup
#      torchvision'ın beklediği [x1,y1,x2,y2] piksel formatına çeviren
#      CocoDetectionDataset sınıfını kullanır (raw baseline ile birebir aynı).
#   3) COCO-pretrained Faster R-CNN (ResNet50-FPN) modelini bu veri setiyle
#      fine-tune eder (early stopping: val mAP50 patience epoch boyunca
#      iyileşmezse durur) — hiperparametreler raw baseline ile AYNI
#      (EPOCHS=30, PATIENCE=8, BATCH_SIZE=2, SGD lr=0.005, StepLR, flip aug,
#      seed=42) ki ablasyonun TEK değişkeni veri seti (crop+CLAHE) olsun.
#   4) En iyi checkpoint ile TEST split üzerinde TEK bir inference geçişinde
#      ham (skorlu, eşiksiz) tahminleri toplar; bunlardan:
#        a) sınıf bazlı (Mass / Suspicious Calcification) ve genel
#           Precision / Recall / mAP50 / mAP50-95 / F1 -> test_metrics.csv
#           (YOLOv8 ve raw baseline ile AYNI format),
#        b) confusion matrix (conf>=0.5, IoU>=0.5, 3x3 background/Mass/Calc,
#           satır bazlı normalize) -> confusion_matrix.png,
#        c) confidence 0.05-0.95 taraması ile sınıf bazlı PR eğrisi
#           -> pr_curve.png
#      hesaplanır.
#
# ÇIKTI YAPISI (raw baseline ile birebir aynı kural, farklı RUN_NAME):
#   <PROJECT_ROOT>/runs/fasterrcnn_crop_clahe/
#     ├── train/
#     │     ├── weights/best.pt, weights/last.pt
#     │     ├── train_log.csv          <- epoch, train_loss, val_mAP50, val_mAP50-95, lr
#     │     └── results.png            <- loss + val mAP eğrileri
#     ├── test_eval/
#     │     ├── per_class_metrics.png   <- P/R/F1/mAP50/mAP50-95 bar grafiği
#     │     ├── confusion_matrix.png    <- bu script tarafından üretilir (ayrı eval scripti YOK)
#     │     └── pr_curve.png            <- bu script tarafından üretilir (ayrı eval scripti YOK)
#     ├── test_metrics.csv             <- raw baseline ile aynı format (model/class/P/R/mAP50/mAP50-95/F1)
#     └── summary.json
#
# NEDEN BU AYARLAR? -> bkz. docs/FASTERRCNN_BASELINE_METHODOLOGY.md
# (Tüm gerekçeler aynen geçerli; bu script sadece DATASET_VARIANT/RUN_NAME
# değiştirilmiş VE confusion matrix/PR eğrisi entegre edilmiş bir ablasyon
# koşusudur.)
# ============================================================

from google.colab import drive
drive.mount("/content/drive")

!pip install -q torchmetrics pycocotools

from pathlib import Path
import json
import random
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.utils.data
import torchvision
from torchvision.transforms import functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pycocotools.coco import COCO

# ------------------------------------------------------------
# 1. Yollar (raw baseline scriptiyle aynı kurallar)
# ------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = Path("/content/drive/MyDrive/vindr_mammo")
DATASET_BASE = PROJECT_ROOT / "dataset"
PREPARED_BASE = DATASET_BASE / "prepared_datasets"

SUBSET_NAME = "target2class_subset_v2_medium_balanced"
DATASET_VARIANT = "crop_clahe_2class"

DRIVE_DATASET_DIR = PREPARED_BASE / f"{SUBSET_NAME}_{DATASET_VARIANT}"
LOCAL_DATASET_DIR = Path(f"/content/vindr_prepared_datasets/{SUBSET_NAME}_{DATASET_VARIANT}")

RUNS_DIR = PROJECT_ROOT / "runs"
RUN_NAME = "fasterrcnn_crop_clahe"
RUN_DIR = RUNS_DIR / RUN_NAME

TRAIN_DIR = RUN_DIR / "train"
TEST_EVAL_DIR = RUN_DIR / "test_eval"
WEIGHTS_DIR = TRAIN_DIR / "weights"
for d in (TRAIN_DIR, TEST_EVAL_DIR, WEIGHTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

print("Drive dataset dir:", DRIVE_DATASET_DIR, "exists:", DRIVE_DATASET_DIR.exists())

# ------------------------------------------------------------
# 2. Veri setini Drive -> local kopyala
# ------------------------------------------------------------

if LOCAL_DATASET_DIR.exists():
    print("Local dataset zaten var, kopyalama atlanıyor:", LOCAL_DATASET_DIR)
else:
    print("Drive -> local kopyalanıyor...")
    shutil.copytree(DRIVE_DATASET_DIR, LOCAL_DATASET_DIR)
    print("Kopyalama tamamlandı:", LOCAL_DATASET_DIR)

# COCO formatındaki annotations/instances_*.json dosyalarında category_id
# zaten 1=Mass, 2=Suspicious Calcification olarak tanımlı (0 torchvision'da
# "background" için ayrılmış durumda) — crop+CLAHE veri seti de aynı
# annotations şemasını kullanıyor (sadece görüntüler crop+CLAHE'li).
CLASS_NAMES = {1: "Mass", 2: "Suspicious Calcification"}
NUM_CLASSES = len(CLASS_NAMES) + 1  # +1 = background

# ------------------------------------------------------------
# 3. Dataset sınıfı: hazır COCO JSON etiketlerini torchvision formatına çevirir
#    (train_fasterrcnn_raw_baseline.py'deki CocoDetectionDataset ile birebir aynı)
# ------------------------------------------------------------


class CocoDetectionDataset(torch.utils.data.Dataset):
    """annotations/instances_<split>.json (COCO formatı) + images/<split>/*.png
    okur, torchvision detection modellerinin beklediği
    {boxes: [x1,y1,x2,y2] (piksel), labels: int64} formatına çevirir.

    COCO bbox formatı [x, y, w, h] (sol-üst köşe + genişlik/yükseklik) —
    burada [x1,y1,x2,y2]'ye çevriliyor. category_id'ler (1=Mass,
    2=Suspicious Calcification) doğrudan torchvision label'ı olarak
    kullanılıyor.

    Görüntüler ORİJİNAL boyutunda döndürülür — resize işlemini model
    (GeneralizedRCNNTransform) kendi içinde yapar ve bbox'ları otomatik
    olarak ölçekler. Bu yüzden burada manuel resize YOK.
    """

    def __init__(self, dataset_dir, split, augment=False):
        self.dataset_dir = Path(dataset_dir)
        ann_file = self.dataset_dir / "annotations" / f"instances_{split}.json"
        self.coco = COCO(str(ann_file))
        self.image_ids = sorted(self.coco.imgs.keys())
        self.augment = augment

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        # img_info["file_name"] dataset köküne göre relatif, örn. "images/test/xxx.png"
        img_path = self.dataset_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            boxes.append([x, y, x + bw, y + bh])
            labels.append(int(ann["category_id"]))

        image = TF.to_tensor(image)

        # Basit augmentasyon: yatay flip (p=0.5) — YOLOv8 scriptindeki
        # fliplr=0.5 ile tutarlı, sadece eğitim setinde uygulanır.
        if self.augment and random.random() < 0.5 and boxes:
            image = image.flip(-1)
            new_boxes = []
            for x1, y1, x2, y2 in boxes:
                new_boxes.append([w - x2, y1, w - x1, y2])
            boxes = new_boxes
        elif self.augment and random.random() < 0.5:
            image = image.flip(-1)

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([img_id]),
            "area": area,
            "iscrowd": torch.zeros((len(labels_t),), dtype=torch.int64),
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


train_ds = CocoDetectionDataset(LOCAL_DATASET_DIR, "train", augment=True)
val_ds = CocoDetectionDataset(LOCAL_DATASET_DIR, "val", augment=False)
test_ds = CocoDetectionDataset(LOCAL_DATASET_DIR, "test", augment=False)

print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

# Batch size = 2: Faster R-CNN + FPN, T4 (16GB) GPU için güvenli değer
# (raw baseline ile aynı). OOM olursa BATCH_SIZE=1 yapın.
BATCH_SIZE = 2
NUM_WORKERS = 2

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, collate_fn=collate_fn,
)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, collate_fn=collate_fn,
)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, collate_fn=collate_fn,
)

# ------------------------------------------------------------
# 4. Model: COCO-pretrained Faster R-CNN (ResNet50-FPN)
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# Sınıflandırma başlığını (box_predictor) bizim sınıf sayımıza göre değiştir
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model.to(device)

# ------------------------------------------------------------
# 5. Eğitim döngüsü (early stopping: val mAP50, patience epoch)
#    -> Ablasyonun TEK değişkeni veri seti olsun diye raw baseline ile
#       AYNI hiperparametreler kullanılıyor (EPOCHS/PATIENCE/optimizer/scheduler).
# ------------------------------------------------------------

EPOCHS = 30
PATIENCE = 8

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0.0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(1, len(data_loader))


@torch.no_grad()
def evaluate_map(model, data_loader, device):
    """torchmetrics ile mAP50 / mAP50-95 hesaplar (COCO-style IoU eşleştirme)."""
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=False)
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        outputs = [{k: v.cpu() for k, v in o.items()} for o in outputs]
        targets_cpu = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
        metric.update(outputs, targets_cpu)
    result = metric.compute()
    return float(result["map_50"]), float(result["map"])


print("\n=== Eğitim başlıyor ===")
log_rows = []
best_map50 = -1.0
epochs_without_improvement = 0

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, optimizer, train_loader, device)
    val_map50, val_map = evaluate_map(model, val_loader, device)
    lr_scheduler.step()

    log_rows.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_mAP50": val_map50,
        "val_mAP50-95": val_map,
        "lr": optimizer.param_groups[0]["lr"],
    })
    print(f"epoch {epoch:3d}/{EPOCHS}  train_loss={train_loss:.4f}  "
          f"val_mAP50={val_map50:.4f}  val_mAP50-95={val_map:.4f}")

    torch.save(model.state_dict(), WEIGHTS_DIR / "last.pt")

    if val_map50 > best_map50:
        best_map50 = val_map50
        epochs_without_improvement = 0
        torch.save(model.state_dict(), WEIGHTS_DIR / "best.pt")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping: val_mAP50 {PATIENCE} epoch boyunca iyileşmedi "
                  f"(en iyi={best_map50:.4f}, epoch {epoch - epochs_without_improvement}).")
            break

log_df = pd.DataFrame(log_rows)
log_df.to_csv(TRAIN_DIR / "train_log.csv", index=False)

# Eğitim eğrileri
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(log_df["epoch"], log_df["train_loss"], marker="o")
axes[0].set_title("Train Loss")
axes[0].set_xlabel("epoch")
axes[1].plot(log_df["epoch"], log_df["val_mAP50"], marker="o", label="val mAP50")
axes[1].plot(log_df["epoch"], log_df["val_mAP50-95"], marker="o", label="val mAP50-95")
axes[1].set_title("Validation mAP")
axes[1].set_xlabel("epoch")
axes[1].legend()
plt.tight_layout()
plt.savefig(TRAIN_DIR / "results.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nKaydedildi:", TRAIN_DIR / "results.png")

# ------------------------------------------------------------
# 6. TEST split üzerinde değerlendirme (en iyi checkpoint ile)
# ------------------------------------------------------------

model.load_state_dict(torch.load(WEIGHTS_DIR / "best.pt"))
model.eval()

# ------------------------------------------------------------
# Greedy eşleştirme yardımcı fonksiyonu (confusion matrix + PR eğrisi için)
#
# Confidence'a göre sıralı tahminleri, IoU>=iou_threshold ile en iyi eşleşen
# (henüz eşleşmemiş) GT kutusuna atar. Çıktı: her biri (gt_label, pred_label)
# çifti — 0 = background:
#   (c, c)   -> doğru tespit (TP, sınıf c)
#   (c, 0)   -> kaçırılan GT (FN, sınıf c)
#   (0, c)   -> yanlış pozitif (FP, sınıf c)
#   (c1, c2) -> sınıf karışıklığı (c1 != c2): c1 için FN, c2 için FP
#
# NOT: evaluate_fasterrcnn.py'deki ile birebir aynı fonksiyon — bu, sınıflar
# arası rekabeti de hesaba katan "global" bir eşleştirme (confidence-sıralı
# greedy IoU eşleştirmesi).
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


@torch.no_grad()
def evaluate_full(model, data_loader, device, class_ids, conf_threshold=0.5, iou_threshold=0.5):
    """Per-class mAP50 / mAP50-95 (torchmetrics) + Precision/Recall/F1
    (conf_threshold'da, IoU>=iou_threshold greedy eşleştirme) hesaplar.

    NOT: Ultralytics, P/R/F1 değerlerini PR eğrisindeki "en iyi F1" noktasında
    raporlar (eşik dinamik). Burada basitlik ve tekrarlanabilirlik için SABİT
    conf_threshold=0.5 kullanıyoruz — bu, P/R/F1 sayılarının YOLO ile birebir
    aynı eşikte olmadığı anlamına gelir. mAP50 / mAP50-95 (asıl karşılaştırma
    metrikleri) ise eşikten bağımsızdır ve tüm modeller için aynı COCO-style
    algoritma (torchmetrics) ile hesaplanır — bu yüzden modeller arası ana
    karşılaştırma mAP50 / mAP50-95 üzerinden yapılmalıdır.
    """
    # İki ayrı metrik: biri sadece IoU=0.5'te (-> mAP50 per class), biri
    # standart COCO IoU=0.5:0.95 aralığında (-> mAP50-95 per class).
    # torchmetrics'in tek bir çağrıda doğrudan "map_50_per_class" döndürmemesi
    # nedeniyle bu ayrım gerekiyor.
    metric_50 = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5], class_metrics=True)
    metric_full = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    tp = {c: 0 for c in class_ids}
    fp = {c: 0 for c in class_ids}
    fn = {c: 0 for c in class_ids}

    # Confusion matrix / PR eğrisi için ham (skorlu, eşiksiz) tahminleri ve
    # GT'leri de aynı geçişte topluyoruz -> ayrı bir inference geçişi GEREKMİYOR.
    all_preds, all_targets = [], []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        outputs_cpu = [{k: v.cpu() for k, v in o.items()} for o in outputs]
        targets_cpu = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
        metric_50.update(outputs_cpu, targets_cpu)
        metric_full.update(outputs_cpu, targets_cpu)

        for out, tgt in zip(outputs_cpu, targets_cpu):
            all_preds.append({"boxes": out["boxes"], "scores": out["scores"], "labels": out["labels"]})
            all_targets.append({"boxes": tgt["boxes"], "labels": tgt["labels"]})

        for out, tgt in zip(outputs_cpu, targets_cpu):
            keep = out["scores"] >= conf_threshold
            pred_boxes = out["boxes"][keep]
            pred_labels = out["labels"][keep]
            gt_boxes = tgt["boxes"]
            gt_labels = tgt["labels"]

            for c in class_ids:
                p_boxes = pred_boxes[pred_labels == c]
                g_boxes = gt_boxes[gt_labels == c]
                matched = set()
                for pb in p_boxes:
                    if len(g_boxes) == 0:
                        fp[c] += 1
                        continue
                    ious = box_iou(pb.unsqueeze(0), g_boxes)[0]
                    best_iou, best_idx = ious.max(0)
                    best_idx = int(best_idx)
                    if float(best_iou) >= iou_threshold and best_idx not in matched:
                        tp[c] += 1
                        matched.add(best_idx)
                    else:
                        fp[c] += 1
                fn[c] += len(g_boxes) - len(matched)

    map50_result = metric_50.compute()
    map_result = metric_full.compute()
    # map_per_class sırası: torchmetrics, görülen sınıf etiketlerini sıralı
    # (artan) sırada döndürür. class_ids zaten [1, 2] = [Mass, Calc] olarak
    # sıralı, dolayısıyla indeks eşlemesi doğrudan uyumlu.
    map50_per_class = map50_result["map_per_class"]
    map_per_class = map_result["map_per_class"]

    rows = []
    sorted_class_ids = sorted(class_ids)
    for i, c in enumerate(sorted_class_ids):
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        rows.append({
            "model": "FasterRCNN-ResNet50-FPN",
            "dataset_variant": DATASET_VARIANT,
            "class": CLASS_NAMES[c],
            "precision": p,
            "recall": r,
            "mAP50": float(map50_per_class[i]),
            "mAP50-95": float(map_per_class[i]),
            "f1": f1,
        })

    # Genel (all) satırı — TP/FP/FN'leri toplayarak (micro-average) P/R/F1,
    # mAP50/mAP50-95 için torchmetrics'in genel "map" değerleri
    # (metric_50.compute()["map"] == mAP@0.5 overall, metric_full -> mAP@0.5:0.95 overall).
    tp_all, fp_all, fn_all = sum(tp.values()), sum(fp.values()), sum(fn.values())
    p_all = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
    r_all = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
    f1_all = 2 * p_all * r_all / (p_all + r_all) if (p_all + r_all) > 0 else 0.0
    rows.append({
        "model": "FasterRCNN-ResNet50-FPN",
        "dataset_variant": DATASET_VARIANT,
        "class": "all",
        "precision": p_all,
        "recall": r_all,
        "mAP50": float(map50_result["map"]),
        "mAP50-95": float(map_result["map"]),
        "f1": f1_all,
    })

    return pd.DataFrame(rows), all_preds, all_targets


results_df, all_preds, all_targets = evaluate_full(model, test_loader, device, class_ids=sorted(CLASS_NAMES.keys()))
print("\n", results_df)

out_csv = RUN_DIR / "test_metrics.csv"
results_df.to_csv(out_csv, index=False)
print("\nKaydedildi:", out_csv)

# Per-class metrik bar grafiği
fig, ax = plt.subplots(figsize=(8, 5))
metrics_to_plot = ["precision", "recall", "mAP50", "mAP50-95", "f1"]
class_rows = results_df[results_df["class"] != "all"]
x = np.arange(len(metrics_to_plot))
width = 0.35
for i, (_, row) in enumerate(class_rows.iterrows()):
    ax.bar(x + i * width, [row[m] for m in metrics_to_plot], width, label=row["class"])
ax.set_xticks(x + width / 2)
ax.set_xticklabels(metrics_to_plot)
ax.set_ylim(0, 1)
ax.set_title("Faster R-CNN (Crop+CLAHE) — Test Split Sınıf Bazlı Metrikler")
ax.legend()
plt.tight_layout()
plt.savefig(TEST_EVAL_DIR / "per_class_metrics.png", dpi=130, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------
# 7. Confusion Matrix (conf>=0.5, IoU>=0.5), satır bazlı normalize
#    -> evaluate_fasterrcnn.py'deki ile aynı mantık, all_preds/all_targets
#       evaluate_full() içindeki TEK inference geçişinden geliyor.
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
ax.set_title(f"Faster R-CNN (Crop+CLAHE) — Confusion Matrix (normalize)\nconf>={CONF_THRESHOLD}, IoU>={IOU_THRESHOLD}")
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
# 8. PR Eğrisi — confidence eşiğini 0.05-0.95 arasında tarayarak
#    sınıf bazlı precision/recall hesapla
# ------------------------------------------------------------

thresholds = np.linspace(0.05, 0.95, 19)
pr_points = {c: {"precision": [], "recall": []} for c in CLASS_NAMES}

for t in thresholds:
    tp_pr = {c: 0 for c in CLASS_NAMES}
    fp_pr = {c: 0 for c in CLASS_NAMES}
    fn_pr = {c: 0 for c in CLASS_NAMES}
    for pred, tgt in zip(all_preds, all_targets):
        for gt_l, pred_l in match_image(pred, tgt, float(t), IOU_THRESHOLD):
            if gt_l == pred_l and gt_l != 0:
                tp_pr[gt_l] += 1
            else:
                if gt_l != 0:
                    fn_pr[gt_l] += 1
                if pred_l != 0:
                    fp_pr[pred_l] += 1
    for c in CLASS_NAMES:
        p = tp_pr[c] / (tp_pr[c] + fp_pr[c]) if (tp_pr[c] + fp_pr[c]) > 0 else 1.0
        r = tp_pr[c] / (tp_pr[c] + fn_pr[c]) if (tp_pr[c] + fn_pr[c]) > 0 else 0.0
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
ax.set_title(f"Faster R-CNN (Crop+CLAHE) — PR Eğrisi (Test, IoU>={IOU_THRESHOLD})")
ax.legend()
plt.tight_layout()
plt.savefig(TEST_EVAL_DIR / "pr_curve.png", dpi=130, bbox_inches="tight")
plt.close()
print("Kaydedildi:", TEST_EVAL_DIR / "pr_curve.png")

# summary.json
summary = {
    "model": "FasterRCNN-ResNet50-FPN",
    "dataset_variant": DATASET_VARIANT,
    "best_weights": str(WEIGHTS_DIR / "best.pt"),
    "epochs_trained": int(log_df["epoch"].max()),
    "best_val_mAP50": best_map50,
    "conf_threshold_for_pr_f1": 0.5,
    "iou_threshold_for_pr_f1": 0.5,
}
with open(RUN_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nTamamlandı. Bu modele ait tüm çıktılar:")
print(" ", RUN_DIR)
print("  ├── train/         (weights/best.pt, weights/last.pt, train_log.csv, results.png)")
print("  ├── test_eval/      (per_class_metrics.png, confusion_matrix.png, pr_curve.png)")
print("  ├── test_metrics.csv")
print("  └── summary.json")
