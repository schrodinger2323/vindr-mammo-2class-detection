# ============================================================
# Ortak Değerlendirme / Karşılaştırma Scripti
#
# Bu script Colab GEREKTİRMEZ — yerelde (proje klasöründe) çalışır.
#
# NE YAPAR?
#   1) runs/<model_run_name>/test_metrics.csv dosyalarının HEPSİNİ bulur
#      (her model scripti aynı kolon formatında üretiyor:
#       model, dataset_variant, class, precision, recall, mAP50, mAP50-95, f1).
#   2) Hepsini tek bir tabloda birleştirir -> runs/comparison/all_models_test_metrics.csv
#   3) Her metrik için (precision, recall, mAP50, mAP50-95, f1) sınıf bazlı
#      (Mass / Suspicious Calcification / all) gruplu bar grafiği üretir
#      -> runs/comparison/comparison_<metrik>.png
#      (gruplar = "model (dataset_variant)" etiketi — aynı model birden
#      fazla veri seti varyantıyla görünüyorsa varyant adı etikete eklenir,
#      örn. "FasterRCNN-ResNet50-FPN (raw_png_2class)" vs
#      "... (crop_clahe_2class)").
#   4) Crop+CLAHE ablasyonu için AYRICA: Faster R-CNN'in raw vs crop+CLAHE
#      sonuçlarını yan yana karşılaştıran özel bir grafik üretir
#      -> runs/comparison/ablation_fasterrcnn_crop_clahe.png
#
# NEDEN?
#   Hocanın istediği "karşılaştırılan yöntemler" (YOLOv8 / Faster R-CNN /
#   RetinaNet) için tek bakışta okunabilir bir özet tablo + görsel üretmek.
#   Yeni bir model eklendiğinde (örn. RetinaNet) bu script otomatik olarak
#   onu da tabloya/grafiklere dahil eder — tek yapılması gereken
#   runs/<yeni_model>/test_metrics.csv dosyasının var olması.
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
OUT_DIR = RUNS_DIR / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1) Tüm test_metrics.csv dosyalarını bul ve birleştir
# ------------------------------------------------------------

csv_files = sorted(RUNS_DIR.glob("*/test_metrics.csv"))
if not csv_files:
    raise FileNotFoundError(f"runs/*/test_metrics.csv bulunamadı (RUNS_DIR={RUNS_DIR})")

print("Bulunan sonuç dosyaları:")
for f in csv_files:
    print(" -", f.relative_to(PROJECT_ROOT))

all_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

out_csv = OUT_DIR / "all_models_test_metrics.csv"
all_df.to_csv(out_csv, index=False)
print("\nKaydedildi:", out_csv.relative_to(PROJECT_ROOT))
print("\n", all_df.to_string(index=False))

# ------------------------------------------------------------
# 2) Etiketleme: aynı model adı (örn. FasterRCNN-ResNet50-FPN) birden fazla
#    dataset_variant ile görünüyorsa, varyant adını etikete ekleyerek
#    pivot'taki çakışmayı (duplicate index) önlüyoruz.
# ------------------------------------------------------------

variant_counts = all_df.groupby("model")["dataset_variant"].nunique()
multi_variant_models = set(variant_counts[variant_counts > 1].index)


def make_label(row):
    if row["model"] in multi_variant_models:
        return f"{row['model']} ({row['dataset_variant']})"
    return row["model"]


all_df["model_label"] = all_df.apply(make_label, axis=1)

# ------------------------------------------------------------
# 3) Her metrik için sınıf bazlı (Mass / Suspicious Calcification / all)
#    gruplu bar grafiği — gruplar = model_label
# ------------------------------------------------------------

CLASS_ORDER = ["Mass", "Suspicious Calcification", "all"]
METRICS = ["precision", "recall", "mAP50", "mAP50-95", "f1"]

labels = list(all_df["model_label"].unique())
n_labels = len(labels)
width = 0.8 / max(n_labels, 1)

for metric in METRICS:
    pivot = all_df.pivot(index="class", columns="model_label", values=metric)
    pivot = pivot.reindex(CLASS_ORDER)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(CLASS_ORDER))
    for i, label in enumerate(labels):
        offsets = [xi + (i - (n_labels - 1) / 2) * width for xi in x]
        ax.bar(offsets, pivot[label], width, label=label)

    ax.set_xticks(list(x))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylim(0, 1)
    ax.set_title(f"Model Karşılaştırması — {metric}")
    ax.set_ylabel(metric)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_png = OUT_DIR / f"comparison_{metric.replace('/', '-')}.png"
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()
    print("Kaydedildi:", out_png.relative_to(PROJECT_ROOT))

# ------------------------------------------------------------
# 4) Crop+CLAHE ablasyonu: Faster R-CNN raw vs crop+CLAHE (varsa)
#    -> tek figürde tüm metrikler, sınıf bazlı (Mass / Calc / all)
# ------------------------------------------------------------

ablation_df = all_df[all_df["model"] == "FasterRCNN-ResNet50-FPN"]
ablation_variants = sorted(ablation_df["dataset_variant"].unique())

if len(ablation_variants) >= 2 and "raw_png_2class" in ablation_variants and "crop_clahe_2class" in ablation_variants:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    variant_labels = {"raw_png_2class": "Raw", "crop_clahe_2class": "Crop+CLAHE"}
    colors = {"raw_png_2class": "#9aa5b1", "crop_clahe_2class": "#2e7d32"}

    for ax, cls in zip(axes, CLASS_ORDER):
        cls_df = ablation_df[ablation_df["class"] == cls]
        x = range(len(METRICS))
        width = 0.35
        for i, variant in enumerate(["raw_png_2class", "crop_clahe_2class"]):
            row = cls_df[cls_df["dataset_variant"] == variant]
            values = [float(row[m].iloc[0]) if not row.empty else 0.0 for m in METRICS]
            offsets = [xi + (i - 0.5) * width for xi in x]
            ax.bar(offsets, values, width, label=variant_labels[variant], color=colors[variant])
        ax.set_xticks(list(x))
        ax.set_xticklabels(METRICS, rotation=20)
        ax.set_ylim(0, 1)
        ax.set_title(cls)

    axes[0].set_ylabel("Değer")
    axes[-1].legend()
    fig.suptitle("Faster R-CNN — Raw vs Crop+CLAHE Ablasyonu (Test Split)")
    plt.tight_layout()
    out_png = OUT_DIR / "ablation_fasterrcnn_crop_clahe.png"
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()
    print("Kaydedildi:", out_png.relative_to(PROJECT_ROOT))
else:
    print("\n(Crop+CLAHE ablasyon grafiği atlandı: raw + crop_clahe_2class için "
          "FasterRCNN-ResNet50-FPN sonuçları henüz mevcut değil.)")

print("\nTamamlandı. Çıktılar:")
print(" ", OUT_DIR.relative_to(PROJECT_ROOT))
print("  ├── all_models_test_metrics.csv")
for metric in METRICS:
    print(f"  ├── comparison_{metric.replace('/', '-')}.png")
print("  └── ablation_fasterrcnn_crop_clahe.png")
