# ============================================================
# YOLOv8 — Raw PNG 2-Class Baseline (Mass + Suspicious Calcification)
#
# Bu script Google Colab'da (GPU runtime) çalıştırılmak üzere tasarlanmıştır.
#
# NE YAPAR?
#   1) target2class_subset_v2_medium_balanced_raw_png_2class veri setini
#      Drive'dan Colab'in lokal diskine kopyalar (eğitim hızı için).
#   2) COCO-pretrained YOLOv8s modelini bu veri setiyle fine-tune eder.
#   3) Eğitim sonunda en iyi checkpoint ile TEST split üzerinde
#      sınıf bazlı (Mass / Suspicious Calcification) ve genel
#      Precision / Recall / mAP50 / mAP50-95 / F1 metriklerini hesaplar.
#   4) Sonuçları CSV olarak kaydeder — bu tablo daha sonra Faster R-CNN ve
#      RetinaNet ile karşılaştırma için kullanılacak ortak format.
#
# ÇIKTI YAPISI (bu modele ait HER ŞEY tek klasör altında):
#   <PROJECT_ROOT>/runs/yolov8s_raw_baseline/
#     ├── train/            <- ağırlıklar, confusion matrix, PR/F1 eğrileri, results.csv
#     ├── test_eval/         <- test split confusion matrix / PR eğrileri
#     ├── test_metrics.csv   <- sınıf bazlı + genel P/R/mAP50/mAP50-95/F1
#     └── summary.json       <- inference hızı, en iyi ağırlık yolu, vb.
#
# NEDEN BU AYARLAR? -> bkz. docs/YOLOV8_BASELINE_METHODOLOGY.md
# ============================================================

from google.colab import drive
drive.mount("/content/drive")


from pathlib import Path
import shutil
import json
import random

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. Yollar
# ------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path("/content/drive/MyDrive/vindr_mammo")
DATASET_BASE = PROJECT_ROOT / "dataset"
PREPARED_BASE = DATASET_BASE / "prepared_datasets"

SUBSET_NAME = "target2class_subset_v2_medium_balanced"
DATASET_VARIANT = "raw_png_2class"   # bu script RAW veri seti için

DRIVE_DATASET_DIR = PREPARED_BASE / f"{SUBSET_NAME}_{DATASET_VARIANT}"

# data.yaml içindeki "path:" alanı bu konuma işaret ediyor (notebook'taki
# LOCAL_*_PREPARED_DIR ile aynı kural). Veri setini buraya kopyalayınca
# data.yaml'da değişiklik yapmaya gerek kalmıyor.
LOCAL_DATASET_DIR = Path(f"/content/vindr_prepared_datasets/{SUBSET_NAME}_{DATASET_VARIANT}")

RUNS_DIR = PROJECT_ROOT / "runs"
RUN_NAME = "yolov8s_raw_baseline"

# Bu modele ait HER ŞEY (eğitim çıktıları, test değerlendirme çıktıları,
# metrik CSV/JSON) bu klasörün altında toplanır.
RUN_DIR = RUNS_DIR / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

print("Drive dataset dir:", DRIVE_DATASET_DIR, "exists:", DRIVE_DATASET_DIR.exists())

# ------------------------------------------------------------
# 2. Veri setini Drive -> local kopyala (I/O hızı için)
# ------------------------------------------------------------

if LOCAL_DATASET_DIR.exists():
    print("Local dataset zaten var, kopyalama atlanıyor:", LOCAL_DATASET_DIR)
else:
    print("Drive -> local kopyalanıyor...")
    shutil.copytree(DRIVE_DATASET_DIR, LOCAL_DATASET_DIR)
    print("Kopyalama tamamlandı:", LOCAL_DATASET_DIR)

DATA_YAML = LOCAL_DATASET_DIR / "data.yaml"
print("data.yaml:", DATA_YAML, "exists:", DATA_YAML.exists())
print(DATA_YAML.read_text())

# ------------------------------------------------------------
# 3. Eğitim
# ------------------------------------------------------------

from ultralytics import YOLO

model = YOLO("yolov8s.pt")  # COCO-pretrained

train_results = model.train(
    data=str(DATA_YAML),
    imgsz=640,
    epochs=100,
    patience=20,          # 20 epoch boyunca val mAP iyileşmezse early stop
    batch=16,              # T4 (16GB) için; OOM olursa 8'e düşür
    optimizer="SGD",       # Ultralytics varsayılanı (lr0=0.01, momentum=0.937)
    seed=SEED,
    project=str(RUN_DIR),
    name="train",
    exist_ok=True,
    plots=True,            # confusion matrix, PR/F1 eğrileri otomatik üretilir
    val=True,
)

best_weights = Path(train_results.save_dir) / "weights" / "best.pt"
print("\nEn iyi ağırlıklar:", best_weights)

# ------------------------------------------------------------
# 4. TEST split üzerinde değerlendirme
# ------------------------------------------------------------

best_model = YOLO(str(best_weights))

test_metrics = best_model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    project=str(RUN_DIR),
    name="test_eval",
    exist_ok=True,
    plots=True,
)

# Sınıf bazlı ve genel metrikleri çıkar
class_names = test_metrics.names  # {0: 'Mass', 1: 'Suspicious Calcification'}

rows = []
for cls_id, cls_name in class_names.items():
    rows.append({
        "model": "YOLOv8s",
        "dataset_variant": DATASET_VARIANT,
        "class": cls_name,
        "precision": float(test_metrics.box.p[cls_id]),
        "recall": float(test_metrics.box.r[cls_id]),
        "mAP50": float(test_metrics.box.ap50[cls_id]),
        "mAP50-95": float(test_metrics.box.ap[cls_id]),
        "f1": float(test_metrics.box.f1[cls_id]),
    })

# Genel (all classes) satırı
rows.append({
    "model": "YOLOv8s",
    "dataset_variant": DATASET_VARIANT,
    "class": "all",
    "precision": float(test_metrics.box.mp),
    "recall": float(test_metrics.box.mr),
    "mAP50": float(test_metrics.box.map50),
    "mAP50-95": float(test_metrics.box.map),
    "f1": float(np.mean(test_metrics.box.f1)),
})

results_df = pd.DataFrame(rows)
print(results_df)

out_csv = RUN_DIR / "test_metrics.csv"
results_df.to_csv(out_csv, index=False)
print("\nKaydedildi:", out_csv)

# Karşılaştırma scriptlerinde kullanmak için ham metrik objesini de json olarak sakla
summary = {
    "model": "YOLOv8s",
    "dataset_variant": DATASET_VARIANT,
    "best_weights": str(best_weights),
    "speed_ms": dict(test_metrics.speed),
    "fitness": float(test_metrics.fitness),
}
with open(RUN_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nTamamlandı. Bu modele ait tüm çıktılar:")
print(" ", RUN_DIR)
print("  ├── train/        (ağırlıklar, confusion matrix, PR/F1 eğrileri, results.csv)")
print("  │    ", train_results.save_dir)
print("  ├── test_eval/     (test split confusion matrix / PR eğrileri)")
print("  │    ", test_metrics.save_dir)
print("  ├── test_metrics.csv")
print("  └── summary.json")
