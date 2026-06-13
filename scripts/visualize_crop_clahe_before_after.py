# ============================================================
# Crop + CLAHE Before/After Görselleştirme
#
# Bu script Google Colab'da çalıştırılmak üzere tasarlanmıştır.
# Gerçek görüntüler (raw PNG ve crop+CLAHE PNG) Google Drive'da
# olduğu için bu görselleştirme yerelde değil, Colab'da üretilir.
#
# Kullanım:
#   1) Drive mount edildikten sonra bu hücreyi/scripti çalıştırın.
#   2) PROJECT_ROOT yolunu kendi Drive yapınıza göre kontrol edin
#      (vindr_mammo_2class_subset_download_2 notebook'u ile aynı).
#   3) Çıktılar:
#        <CROP_DIR>/eda/figures/before_after_bbox_examples.png
#        <CROP_DIR>/eda/figures/clahe_histogram_examples.png
#      Bu dosyaları indirip projeyi yüklediğiniz klasöre kopyalayın.
# ============================================================

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Yollar (notebook ile aynı kurallar)
# ------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path("/content/drive/MyDrive/vindr_mammo")
DATASET_BASE = PROJECT_ROOT / "dataset"
PREPARED_BASE = DATASET_BASE / "prepared_datasets"

SUBSET_NAME = "target2class_subset_v2_medium_balanced"

RAW_DIR = PREPARED_BASE / f"{SUBSET_NAME}_raw_png_2class"
CROP_DIR = PREPARED_BASE / f"{SUBSET_NAME}_crop_clahe_2class"

OUT_FIG_DIR = CROP_DIR / "eda" / "figures"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "Mass", 1: "Susp. Calc."}
# BGR renkler: Mass = kirmizi, Calcification = turuncu
CLASS_COLORS_BGR = {0: (0, 0, 255), 1: (0, 165, 255)}

print("RAW_DIR  :", RAW_DIR, "exists:", RAW_DIR.exists())
print("CROP_DIR :", CROP_DIR, "exists:", CROP_DIR.exists())

# ------------------------------------------------------------
# 2. Yardımcı fonksiyonlar
# ------------------------------------------------------------

def read_yolo_label(path: Path):
    labels = []
    if not path.exists():
        return labels
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])
        labels.append((cls, xc, yc, bw, bh))
    return labels


def draw_boxes(gray_img, labels, thickness=4, font_scale=1.1):
    """Gri görüntü üzerine YOLO formatındaki kutuları çizer, BGR döner."""
    img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    for cls, xc, yc, bw, bh in labels:
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        color = CLASS_COLORS_BGR.get(cls, (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            img, CLASS_NAMES.get(cls, str(cls)), (x1, max(0, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 3
        )
    return img


def to_rgb(bgr_img):
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------
# 3. Örnek seçimi: Mass, Calcification, negatif, fallback
# ------------------------------------------------------------

transform_log = pd.read_csv(CROP_DIR / "logs" / "crop_clahe_transform_log.csv")

mass_candidates = []
calc_candidates = []
negative_candidates = []
fallback_candidates = []

for _, row in transform_log.iterrows():
    split, image_id = row["split"], row["image_id"]
    raw_lab = read_yolo_label(RAW_DIR / "labels" / split / f"{image_id}.txt")
    classes_present = {c for c, *_ in raw_lab}

    if row["crop_status"] != "ok":
        fallback_candidates.append((split, image_id))
    if 0 in classes_present and 1 not in classes_present:
        mass_candidates.append((split, image_id))
    if 1 in classes_present:
        calc_candidates.append((split, image_id))
    if not raw_lab:
        negative_candidates.append((split, image_id))

random.shuffle(mass_candidates)
random.shuffle(calc_candidates)
random.shuffle(negative_candidates)
random.shuffle(fallback_candidates)

N_PER_CATEGORY = 2
selected = (
    [("Mass", s, i) for s, i in mass_candidates[:N_PER_CATEGORY]]
    + [("Suspicious Calcification", s, i) for s, i in calc_candidates[:N_PER_CATEGORY]]
    + [("Negatif (bulgu yok)", s, i) for s, i in negative_candidates[:N_PER_CATEGORY]]
    + [("Fallback (crop yapılmadı)", s, i) for s, i in fallback_candidates[:N_PER_CATEGORY]]
)

print(f"Seçilen örnek sayısı: {len(selected)}")
for cat, split, image_id in selected:
    print(f"  - {cat}: {split}/{image_id}")

# ------------------------------------------------------------
# 4. Before/After bbox overlay grid
# ------------------------------------------------------------

n_rows = len(selected)
fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4.5 * n_rows))
if n_rows == 1:
    axes = axes.reshape(1, 2)

for row_idx, (cat, split, image_id) in enumerate(selected):
    raw_img = cv2.imread(str(RAW_DIR / "images" / split / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
    crop_img = cv2.imread(str(CROP_DIR / "images" / split / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)

    raw_lab = read_yolo_label(RAW_DIR / "labels" / split / f"{image_id}.txt")
    crop_lab = read_yolo_label(CROP_DIR / "labels" / split / f"{image_id}.txt")

    raw_overlay = to_rgb(draw_boxes(raw_img, raw_lab))
    crop_overlay = to_rgb(draw_boxes(crop_img, crop_lab))

    axes[row_idx, 0].imshow(raw_overlay)
    axes[row_idx, 0].set_title(f"[{cat}] RAW — {split}/{image_id}\n{raw_img.shape[1]}x{raw_img.shape[0]}")
    axes[row_idx, 0].axis("off")

    axes[row_idx, 1].imshow(crop_overlay)
    axes[row_idx, 1].set_title(f"[{cat}] CROP+CLAHE — {split}/{image_id}\n{crop_img.shape[1]}x{crop_img.shape[0]}")
    axes[row_idx, 1].axis("off")

plt.tight_layout()
out_path = OUT_FIG_DIR / "before_after_bbox_examples.png"
plt.savefig(out_path, dpi=130, bbox_inches="tight")
plt.close()
print("Kaydedildi:", out_path)

# ------------------------------------------------------------
# 5. CLAHE histogram karşılaştırması
#    (raw crop bölgesi vs CLAHE uygulanmış aynı bölge)
# ------------------------------------------------------------

hist_examples = (mass_candidates[:1] + calc_candidates[:1] + negative_candidates[:1])
hist_examples = [(s, i) for s, i in hist_examples]

n_rows = len(hist_examples)
fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.5 * n_rows))
if n_rows == 1:
    axes = axes.reshape(1, 3)

for row_idx, (split, image_id) in enumerate(hist_examples):
    raw_img = cv2.imread(str(RAW_DIR / "images" / split / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
    crop_img = cv2.imread(str(CROP_DIR / "images" / split / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)

    log_row = transform_log[(transform_log.split == split) & (transform_log.image_id == image_id)].iloc[0]
    x1, y1, x2, y2 = int(log_row.crop_x1), int(log_row.crop_y1), int(log_row.crop_x2), int(log_row.crop_y2)
    raw_crop_region = raw_img[y1:y2, x1:x2]  # CLAHE öncesi, aynı crop bölgesi

    axes[row_idx, 0].imshow(raw_crop_region, cmap="gray")
    axes[row_idx, 0].set_title(f"Crop bölgesi (CLAHE öncesi)\n{split}/{image_id}")
    axes[row_idx, 0].axis("off")

    axes[row_idx, 1].imshow(crop_img, cmap="gray")
    axes[row_idx, 1].set_title(f"Crop + CLAHE (sonrası)\n{split}/{image_id}")
    axes[row_idx, 1].axis("off")

    axes[row_idx, 2].hist(raw_crop_region.ravel(), bins=64, range=(0, 255), alpha=0.6, label="CLAHE öncesi", color="#4C72B0")
    axes[row_idx, 2].hist(crop_img.ravel(), bins=64, range=(0, 255), alpha=0.6, label="CLAHE sonrası", color="#C44E52")
    axes[row_idx, 2].set_title("Piksel yoğunluk histogramı")
    axes[row_idx, 2].legend()

plt.tight_layout()
out_path = OUT_FIG_DIR / "clahe_histogram_examples.png"
plt.savefig(out_path, dpi=130, bbox_inches="tight")
plt.close()
print("Kaydedildi:", out_path)

print("\nTamamlandı. Figürleri Drive'dan indirip proje klasörünüzdeki")
print(f"dataset/prepared_datasets/{SUBSET_NAME}_crop_clahe_2class/eda/figures/ altına ekleyin.")
