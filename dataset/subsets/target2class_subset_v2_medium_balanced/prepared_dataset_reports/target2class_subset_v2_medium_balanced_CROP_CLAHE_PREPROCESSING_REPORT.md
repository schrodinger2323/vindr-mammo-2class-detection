# target2class_subset_v2_medium_balanced — Crop + CLAHE Preprocessing Report

## Amaç

Bu rapor, `target2class_subset_v2_medium_balanced_raw_png_2class` veri setinden üretilen
`target2class_subset_v2_medium_balanced_crop_clahe_2class` veri setinin oluşturulma
sürecini, kullanılan algoritmayı ve elde edilen istatistikleri özetler.

Bu adım, literatürde (örn. Abdikenov et al. 2025, *J. Imaging*) VinDr-Mammo üzerinde
mAP50'yi 0.438 → 0.590'a (p=0.008) çıkardığı gösterilen "meme bölgesi crop + CLAHE
kontrast iyileştirme" ön işleme adımının uygulanmasıdır.

## Algoritma

### 1. Meme bölgesi tespiti ve crop (`find_breast_crop_box`)

1. Görüntü Gaussian blur (5x5) ile yumuşatılır.
2. Otsu eşikleme (`cv2.THRESH_OTSU`) ile ikili maske oluşturulur.
3. Maske alanı toplam alanın %5'inden az veya %98'inden fazlaysa (etiket/artefakt
   baskınlığı durumları), persentil tabanlı eşikleme (`gray > percentile(gray, 2)`)
   ile yeniden maske oluşturulur.
4. Morfolojik close + open (9x9 kernel) ile gürültü temizlenir.
5. En büyük kontur bulunur; konturun alanı görüntü alanının %5'inden küçükse
   crop uygulanmaz (fallback: tam görüntü).
6. En büyük konturun bounding box'ına genişlik/yükseklikte %3.5 margin eklenerek
   nihai crop kutusu belirlenir.

Bu yöntem, literatürdeki "merkezi şerit tarama" yaklaşımına alternatif olarak Otsu +
kontur tabanlı, daha standart bir meme bölgesi segmentasyonu kullanır (Karaca Aydemir
et al. 2025'teki "ikili eşikleme ile en büyük meme bölgesini bulma" yaklaşımına yakın).

### 2. CLAHE kontrast iyileştirme (`apply_clahe`)

Crop edilen gri görüntüye OpenCV CLAHE uygulanır:

- `clipLimit = 2.0`
- `tileGridSize = (8, 8)`

Bu parametreler Abdikenov et al. 2025'teki tarifle birebir aynıdır.

### 3. Etiket dönüşümü (`transform_labels_to_crop`)

YOLO formatındaki bbox'lar crop ofsetine göre kaydırılır, crop sınırlarına clip
edilir ve yeni crop boyutlarına göre yeniden normalize edilir. Crop sonrası tamamen
crop alanı dışında kalan kutular düşürülür ve loglanır.

## Veri Seti İstatistikleri

### Görüntü sayıları (raw ile aynı — sadece içerik dönüştürüldü)

| split | image_count |
|---|---|
| train | 1418 |
| val   | 306  |
| test  | 308  |
| **toplam** | **2032** |

### Crop alan oranı (crop_area / orig_area)

| istatistik | değer |
|---|---|
| ortalama | 0.306 |
| medyan | 0.285 |
| std | 0.117 |
| min | 0.100 |
| max | 1.000 |

Ortalama olarak görüntülerin yaklaşık **%31'i** (meme dokusu + margin) korunmuş,
geri kalan kısım (siyah arka plan, etiketler, artefaktlar) kırpılmıştır.
Genişlik oranı ortalama ~0.38, yükseklik oranı ortalama ~0.79 — yani kırpma
ağırlıklı olarak **yatay eksende** gerçekleşmiştir (dikey boyut büyük ölçüde
korunmuştur), bu da meme dokusunun görüntünün bir kenarına yaslı olduğu VinDr-Mammo
mamogramlarıyla tutarlıdır.

### Crop durumu

| durum | görüntü sayısı |
|---|---|
| ok | 2025 |
| fallback_small_contour (crop uygulanmadı, tam görüntü + CLAHE) | 7 |

### Bbox sayıları: Raw vs Crop+CLAHE

| split | Mass (raw) | Mass (crop+CLAHE) | Calcification (raw) | Calcification (crop+CLAHE) |
|---|---|---|---|---|
| train | 346 | 346 | 318 | 315 |
| val   | 77  | 77  | 54  | 54  |
| test  | 69  | 69  | 61  | 58  |

### Düşürülen kutular

Toplam 925 kutudan **6'sı** (%0.65) crop sonrası tamamen crop alanı dışında kaldığı
için düşürüldü — tamamı **Suspicious Calcification** sınıfından (4 görüntüde,
train: 2, test: 4 kutu). Bu, meme kenarına yakın küçük kalsifikasyonların crop
sırasında kaybedilebileceğini gösteriyor; oran düşük olduğu için kabul edilebilir,
ancak raporda bir limitasyon olarak not edilecek.

## Üretilen Figürler

- `eda/figures/crop_transform_summary.png` — crop alan oranı, genişlik/yükseklik
  oranı dağılımları ve crop durumu sayıları (yerel olarak `crop_clahe_transform_log.csv`
  üzerinden üretildi).
- `eda/figures/class_distribution_raw_vs_crop_clahe.png` — split başına sınıf bazlı
  bbox sayısı karşılaştırması (raw vs crop+CLAHE).
- `eda/figures/before_after_bbox_examples.png` *(Colab'da üretilecek)* — örnek
  görüntülerde GT bbox overlay ile raw vs crop+CLAHE karşılaştırması (Mass,
  Calcification, negatif, fallback örnekleri).
- `eda/figures/clahe_histogram_examples.png` *(Colab'da üretilecek)* — CLAHE
  öncesi/sonrası piksel yoğunluk histogram karşılaştırması.

Son iki figür `scripts/visualize_crop_clahe_before_after.py` ile Colab'da
üretilir (gerçek görüntüler Drive'da olduğu için yerelde üretilemiyor).
Üretildikten sonra Drive'dan indirilip bu klasöre eklenmelidir.

## Sonraki Adım

Preprocessing adımı tamamlandı. Sıradaki adım: YOLOv8 ile raw veri seti üzerinde
baseline eğitim (uygulama planının 2. adımı).
