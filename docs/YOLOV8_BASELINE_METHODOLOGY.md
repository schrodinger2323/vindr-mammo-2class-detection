# YOLOv8 Raw Baseline — Metodoloji ve Gerekçeler

Bu doküman `scripts/train_yolov8_raw_baseline.py` scriptindeki tasarım kararlarını
ve bu kararların literatürdeki hangi bulgulara dayandığını açıklar. Amaç: hocanın
istediği "işlem adımlarının ve gerekçelerinin açıkça dokümante edilmesi" şartını
karşılamak.

## 1. Neden YOLOv8 (ve neden "s" varyantı)?

Görev tanımı YOLO'nun herhangi bir versiyonuna izin veriyor. YOLOv8'i seçtik çünkü:

- Ultralytics tarafından aktif sürdürülen, kararlı ve Colab'da kurulumu/eğitimi en
  basit olan versiyon (`pip install ultralytics`, tek satır `model.train()`).
- Literatürde VinDr-Mammo ve ilişkili mamografi veri setlerinde doğrudan
  karşılaştırma noktaları mevcut:
  - Karaca Aydemir et al. (2025) YOLOv5n/s/m'yi karşılaştırmış, **YOLOv5s** en iyi
    sonucu vermiş (VinDr-Mammo → INbreast transfer, mAP50=0.84).
  - Abdikenov et al. (2025), YOLOv8 ile TL kullanarak INbreast'te mAP50=0.873
    (Cao et al. 2024'ten referans) elde edildiğini raporluyor; kendi
    YOLOv12-L/X karşılaştırmalarında da "s" ve "l" varyantları güçlü
    baseline'lar.

"s" (small) varyantını seçtik: "n" (nano)'ya göre daha yüksek kapasite, "m"/"l"'ye
göre Colab'ın ücretsiz T4 GPU'sunda daha hızlı eğitim/iterasyon. Veri setimiz
nispeten küçük (1424 train görüntüsü) olduğu için "s" kapasitesi makul bir
başlangıç noktası; gerekirse "m" ile tekrar denenebilir.

## 2. Neden COCO-pretrained (transfer learning)?

Tüm incelenen makaleler (Abdikenov 2025, Karaca Aydemir 2025, Ribli 2018) COCO
veya benzeri büyük veri setleri üzerinde pretrained backbone kullanıyor ve bunun
küçük tıbbi görüntü veri setlerinde yakınsamayı hızlandırdığını ve performansı
artırdığını gösteriyor. `yolov8s.pt` (COCO-pretrained) ağırlıklarından
fine-tuning yapıyoruz.

## 3. Neden imgsz=640?

- Ultralytics YOLOv8'in varsayılan ve en yaygın test edilen giriş boyutu.
- Abdikenov et al. (2025) crop+CLAHE pipeline'ında görüntüleri 640x640'a
  resize ediyor. Raw baseline'ı da 640 ile eğiterek **raw vs crop+CLAHE
  karşılaştırmasını adil** tutuyoruz (aynı giriş boyutu, sadece preprocessing
  farklı).
- Not: Bizim raw görüntülerimiz büyük (1465x2048 / 1630x2048). YOLO bunları
  640'a letterbox ile resize edecek — küçük kalsifikasyonlar için bu bir
  bilgi kaybı kaynağı olabilir (literatürde de calcification mAP50'nin
  düşük olmasının nedenlerinden biri olarak "yüksek çözünürlüğün 640'a
  düşürülmesi" gösteriliyor). Bu, ileride patch/tile deneyi için bir motivasyon
  olarak rapora not edilecek.

## 4. Eğitim hiperparametreleri

| Parametre | Değer | Gerekçe |
|---|---|---|
| epochs | 100 | Abdikenov et al. (2025) 100 epoch + early stopping kullanıyor. |
| patience | 20 | Abdikenov 10 epoch patience kullanıyor (AdamW+cosine ile); biz Ultralytics varsayılan SGD ile eğittiğimiz için biraz daha toleranslı patience seçtik (20). Gerekirse düşürülebilir. |
| batch | 16 | T4 (16GB) GPU'da 640px + yolov8s için tipik güvenli değer. OOM olursa 8'e düşürülmeli. |
| optimizer | SGD (Ultralytics varsayılanı: lr0=0.01, momentum=0.937, weight_decay=0.0005) | Ultralytics'in YOLOv8 için en çok test edilmiş/ayarlanmış konfigürasyonu — "iyi tanımlanmış baseline" olarak başlangıç noktası. Abdikenov AdamW+1e-4+cosine kullanıyor; bu alternatif, ileride bir ablasyon olarak denenebilir. |
| seed | 42 | Veri seti split'inde kullanılan seed ile tutarlı — tüm pipeline'da tekrarlanabilirlik. |
| augmentation | Ultralytics varsayılanı (mosaic, hsv_h/s/v, fliplr, translate, scale, vb.) | Karaca Aydemir et al.'in "medium" augmentasyon konfigürasyonuna (Tablo 2) yakın; literatürde augmentasyonun mAP50'yi anlamlı şekilde artırdığı gösteriliyor. |

## 5. Sınıf dengesi notu

Literatürdeki çalışmalarda (özellikle kombine veri setlerinde) Mass/Calcification
dengesizliği ciddi sorun yaratıyor (calcification mAP50 < 0.12). Bizim
`medium_balanced` alt kümemizde train split'inde **346 Mass / 318 Calcification**
kutusu var — yani **görece dengeli**. Bu nedenle baseline'da ekstra
class-weighting veya oversampling uygulamıyoruz; ancak test sonuçlarında
calcification performansı düşük çıkarsa (literatürle tutarlı bir şekilde),
bu durum bilinen bir zorluk olarak yorumlanacak, hata olarak değil.

## 6. Değerlendirme

- **Split:** Eğitimde train+val kullanılır (val erken durdurma için izlenir),
  final metrikler **test split (308 görüntü, hiç görülmemiş)** üzerinde
  hesaplanır — patient/study-level split sayesinde veri sızıntısı yok.
- **Metrikler:** Sınıf bazlı (Mass, Suspicious Calcification) ve genel
  Precision, Recall, mAP@0.5, mAP@0.5:0.95, F1 — Abdikenov et al.'in
  Tablo 4-7 formatıyla aynı, böylece Faster R-CNN ve RetinaNet sonuçlarıyla
  doğrudan karşılaştırılabilir bir CSV üretilir.
- **Çıktı klasörü:** Bu modele ait eğitim ve değerlendirme çıktılarının tamamı
  tek bir klasör altında toplanır: `runs/yolov8s_raw_baseline/`
  (`train/`, `test_eval/`, `test_metrics.csv`, `summary.json`). Diğer modeller
  (Faster R-CNN, RetinaNet) için de aynı kural: `runs/<model_run_name>/` altında
  kendi alt klasörleri.
- **Görselleştirme:** `plots=True` ile Ultralytics otomatik olarak confusion
  matrix (jimaging Fig. 5-7,10 ile aynı format) ve PR/F1 eğrilerini üretir.
  GT vs prediction bbox overlay (TP/FP/FN) karşılaştırmaları, tüm modeller
  eğitildikten sonra ortak bir görselleştirme scriptiyle (plan adım 5-6)
  üretilecek.

## 7. Beklenen sonuçlar (literatür kalibrasyonu)

VinDr-Mammo üzerinde raw (preprocessing'siz) YOLO modelleri için literatürde
mAP50 ≈ 0.44 (mass, Abdikenov et al., preprocessing öncesi). Calcification
için mAP50 tipik olarak < 0.05-0.12 aralığında. Bu sayılar, sonuçlarımızı
yorumlarken bir referans noktası olarak kullanılacak — özellikle crop+CLAHE
veri setiyle tekrar eğitildiğinde (plan adım 6) elde edilecek iyileşmeyi
ölçmek için.

## Sonraki adım

Bu script çalıştırıldıktan sonra: Faster R-CNN (torchvision) raw baseline
scripti (plan adım 3).
