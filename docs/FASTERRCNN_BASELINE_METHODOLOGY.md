# Faster R-CNN (ResNet50-FPN) Raw Baseline — Metodoloji ve Gerekçeler

Bu doküman `scripts/train_fasterrcnn_raw_baseline.py` scriptindeki tasarım
kararlarını ve gerekçelerini açıklar — YOLOv8 metodoloji dokümanıyla
(`docs/YOLOV8_BASELINE_METHODOLOGY.md`) aynı amaca hizmet eder: hocanın
istediği "işlem adımlarının ve gerekçelerinin dokümante edilmesi" şartı.

## 1. Neden Faster R-CNN (ve neden ResNet50-FPN)?

Görev tanımı, YOLO ailesine ek olarak iki aşamalı (two-stage) bir dedektörle
karşılaştırma istiyor. Faster R-CNN, mamografi lezyon tespiti literatüründe en
çok kullanılan two-stage mimari (örn. Ribli et al. 2018, s41598-018-22437-z —
VinDr-Mammo'nun da atıfta bulunduğu çalışmalardan biri, Faster R-CNN tabanlı
bir dedektörle mamogramlarda kitle/kalsifikasyon tespiti yapıyor).

`ResNet50-FPN` backbone'unu seçtik çünkü:

- torchvision'da hazır, COCO-pretrained ağırlıklarla birlikte gelen "standart"
  Faster R-CNN konfigürasyonu (`fasterrcnn_resnet50_fpn`).
- FPN (Feature Pyramid Network), farklı boyutlardaki nesneleri (büyük kitleler
  vs. küçük kalsifikasyonlar) çok ölçekli özellik haritalarıyla yakalayabiliyor
  — bu, VinDr-Mammo'daki boyut çeşitliliği için önemli.
- YOLOv8 (tek aşamalı, anchor-free) ile mimari olarak en uzak/karşıt nokta —
  bu da "YOLO ailesi vs. klasik two-stage CNN" karşılaştırmasını anlamlı
  kılıyor.

## 2. Neden COCO-pretrained (transfer learning)?

YOLOv8 baseline'ı ile aynı gerekçe: küçük tıbbi veri setlerinde (1424 train
görüntüsü) ImageNet/COCO-pretrained backbone'lar yakınsamayı hızlandırıyor ve
performansı artırıyor (Abdikenov 2025, Karaca Aydemir 2025, Ribli 2018 — tümü
pretrained backbone kullanıyor). `fasterrcnn_resnet50_fpn(weights="DEFAULT")`
ile COCO-pretrained ağırlıklardan başlıyoruz, sadece sınıflandırma başlığını
(`box_predictor`) 2 sınıf + background = 3 çıkışa göre yeniden ilkliyoruz.

## 3. Veri formatı: hazır COCO JSON kullanımı

VinDr-Mammo alt kümemiz iki paralel etiket formatıyla geliyor: YOLOv8
scriptinin kullandığı YOLO-txt formatı (`labels/<split>/*.txt`, normalize
`class xc yc w h`) ve `annotations/instances_{train,val,test}.json`
(COCO formatı, `bbox=[x,y,w,h]` piksel + `category_id`). İkisi de aynı
temel etiketlerin farklı serileştirmeleri; hangisinin kullanılacağı modelin
"doğal" beklediği formata göre seçildi:

- **YOLOv8** -> `data.yaml` + YOLO-txt etiketleri (Ultralytics'in beklediği
  format, zaten mevcut).
- **Faster R-CNN / RetinaNet (torchvision)** -> hazır **COCO JSON**
  (`pycocotools.coco.COCO` + `CocoDetectionDataset`). Bu seçimin gerekçeleri:
  - torchvision'ın resmi "object detection finetuning" tutorial'ı ve
    COCO-pretrained modellerin "doğal" formatı COCO'dur — bu nedenle
    torchvision tabanlı modeller için **standart pratik** budur.
  - COCO JSON'daki `category_id` tanımı (1=Mass, 2=Suspicious
    Calcification; 0 torchvision'da "background" için zaten ayrılmış)
    bizim ihtiyacımız olan şemayla **birebir aynı** — YOLO-txt'den geçişte
    gerekli olan +1 ID kaydırmasına burada **gerek yok**.
  - Aynı etiketleri YOLO-txt'den yeniden ayrıştırmak yerine hazır ve
    standart bir formatı kullanmak kod tekrarını azaltıyor ve "her model
    kendi doğal formatını kullanıyor" eşleşmesini raporda daha açık hale
    getiriyor.

`CocoDetectionDataset` sınıfının yaptığı dönüşümler:

- `annotations/instances_<split>.json` dosyasını `pycocotools.coco.COCO`
  ile yükler; her görüntü için `images` listesindeki `file_name` alanından
  (örn. `images/test/<id>.png`, dataset köküne göre relatif) dosyayı okur.
- Her annotation için COCO `bbox=[x,y,w,h]` (sol-üst köşe + genişlik/
  yükseklik, piksel) formatını torchvision'ın beklediği `[x1,y1,x2,y2]`'ye
  çevirir (`x2=x+w, y2=y+h`).
- `category_id`'yi (1=Mass, 2=Suspicious Calcification) doğrudan
  `labels` tensörüne yazar — ek bir kaydırma işlemi yok.
- Eğitim setinde yatay flip (p=0.5) augmentasyonu uygulanır (YOLOv8'in
  `fliplr=0.5` ayarıyla tutarlı); flip sırasında kutu koordinatları da
  orantılı olarak güncellenir.
- Görüntüler **orijinal boyutlarında** (resize edilmeden) modele veriliyor —
  `GeneralizedRCNNTransform` (modelin içinde) resize işlemini görüntü VE
  bbox'lar için otomatik ve orantılı şekilde yapıyor. Bu, YOLOv8'in
  `imgsz=640` ile manuel resize yapmasından farklı; Faster R-CNN'in
  varsayılan davranışı `min_size=800, max_size=1333` ile resize etmek
  (torchvision varsayılanı, değiştirilmedi).

**Not (çözünürlük farkı):** YOLOv8 640px, Faster R-CNN ~800px girdi kullanıyor.
Bu, iki mimarinin "kendi standart/varsayılan pratiği" ile çalıştırılması
anlamına geliyor — literatürdeki model karşılaştırmalarında da (örn.
Abdikenov 2025'in farklı YOLO sürümleri vs. CNN karşılaştırmaları) her
mimari kendi tipik girdi boyutuyla değerlendiriliyor. Bu, tam "apples-to-apples"
değil ama her modelin "iyi pratiği"yle değerlendirilmesini sağlıyor; raporda
bir karşılaştırma sınırlaması olarak not edilecek.

## 4. Eğitim hiperparametreleri

| Parametre | Değer | Gerekçe |
|---|---|---|
| epochs | 30 (max), early stopping patience=8 | Faster R-CNN epoch başına YOLOv8'den çok daha yavaş (iki aşamalı: RPN + ROI head). 30 epoch + early stopping, T4 üzerinde makul bir süre içinde yakınsama sağlıyor; YOLOv8'deki patience/epoch oranına (20/100=0.2) yakın bir oran (8/30≈0.27) korunuyor. |
| batch_size | 2 | T4 (16GB) GPU için Faster R-CNN+FPN belleği YOLOv8'den çok daha yüksek (RPN + ROI pooling + büyük girdi boyutu). OOM olursa batch_size=1'e düşürülmeli. |
| optimizer | SGD (lr=0.005, momentum=0.9, weight_decay=0.0005) | torchvision'ın resmi "object detection finetuning" tutorial'ında önerilen standart değerler — Faster R-CNN için en yaygın referans konfigürasyon. |
| lr scheduler | StepLR (step_size=10, gamma=0.1) | 30 epoch'ta 2 kademeli düşüş (epoch 10 ve 20'de lr/10) — torchvision tutorial'ındaki yaklaşımla aynı mantık, daha uzun eğitim süresine ölçeklendirilmiş. |
| augmentation | Yatay flip (p=0.5) | YOLOv8'in `fliplr=0.5` ayarıyla tutarlı, minimal ve karşılaştırılabilir bir augmentasyon. Faster R-CNN için torchvision'da mosaic/HSV gibi YOLO-spesifik augmentasyonlar mevcut değil; ek augmentasyon eklemek modeller arası karşılaştırmayı karmaşıklaştırır. |
| seed | 42 | Veri seti split'i ve YOLOv8 scriptiyle tutarlı. |

## 5. Erken durdurma kriteri

Her epoch sonunda **val split üzerinde mAP@0.5** (torchmetrics ile) hesaplanır.
`PATIENCE=8` epoch boyunca yeni bir en iyi değer elde edilmezse eğitim durur
ve en iyi (`best.pt`) checkpoint test değerlendirmesinde kullanılır — YOLOv8
scriptindeki mantıkla birebir aynı (orada Ultralytics bunu otomatik yapıyor).

## 6. Değerlendirme metrikleri ve önemli bir kısıtlama

- **mAP50 / mAP50-95:** `torchmetrics.detection.MeanAveragePrecision` ile
  COCO-style IoU eşleştirmesiyle hesaplanır — Ultralytics'in YOLOv8 için
  kullandığı algoritmayla aynı aileden (COCO mAP tanımı), bu nedenle
  **modeller arası asıl karşılaştırma bu iki metrik üzerinden yapılmalıdır**.
- **Precision / Recall / F1:** Ultralytics, bu değerleri PR eğrisindeki
  "en iyi F1" noktasında (dinamik eşik) raporluyor. Faster R-CNN
  scriptinde basitlik ve tekrarlanabilirlik için **sabit `confidence>=0.5`,
  `IoU>=0.5`** eşiğinde greedy eşleştirme ile hesaplanıyor. Bu, P/R/F1
  sayılarının YOLO ile *birebir* aynı operasyon noktasında olmadığı anlamına
  gelir — raporda bu fark açıkça belirtilecek, ana karşılaştırma mAP50/mAP50-95
  üzerinden yapılacaktır.
- **Çıktı formatı:** `test_metrics.csv`, YOLOv8 scriptiyle aynı kolonlara
  sahiptir (`model, dataset_variant, class, precision, recall, mAP50,
  mAP50-95, f1`) — sadece `model` kolonu farklı, böylece tüm modellerin
  sonuçları tek bir tabloda birleştirilebilir.
- **Confusion matrix ve PR eğrisi neden ayrı bir scriptte?** Ultralytics
  (YOLOv8), `model.val()` çağrısında `confusion_matrix_normalized.png` ve
  `BoxPR_curve.png`'yi **otomatik** üretiyor — bu, Ultralytics'in dahili
  plot fonksiyonlarından geliyor ve torchvision detection modellerinde
  (Faster R-CNN, RetinaNet) hazır bir karşılığı yok. Bu nedenle
  `scripts/evaluate_fasterrcnn.py` adlı ayrı bir script yazıldı:
  eğitilmiş `best.pt`'yi yükler (yeniden eğitim YOK, sadece inference),
  test split üzerinde tüm tahminleri toplar ve:
  - **Confusion matrix** (`confusion_matrix.png`): `conf>=0.5, IoU>=0.5`
    eşiğinde, background dahil 3x3 (background/Mass/Suspicious
    Calcification), satır bazlı normalize — YOLOv8'in
    `confusion_matrix_normalized.png`'siyle aynı yorumlama mantığı
    ("gerçek sınıf X'in ne kadarı Y olarak tahmin edildi").
  - **PR eğrisi** (`pr_curve.png`): confidence eşiği 0.05-0.95 arasında
    taranarak sınıf bazlı precision/recall noktaları hesaplanır ve
    eğri olarak çizilir — YOLOv8'in `BoxPR_curve.png`'siyle
    karşılaştırılabilir formatta.
  - Bu script `RUN_NAME`/`DATASET_VARIANT` değiştirilerek genel olarak
    yeniden kullanılabilir tasarlandı; ancak crop+CLAHE ablasyonunda
    (adım 6, `train_fasterrcnn_crop_clahe.py`) ayrı bir script çalıştırmamak
    için aynı `match_image()` mantığı doğrudan eğitim scriptinin sonuna
    entegre edildi — test split üzerindeki TEK inference geçişi hem
    `test_metrics.csv` hem de `confusion_matrix.png`/`pr_curve.png` için
    kullanılıyor.

## 7. Beklenen sonuçlar (literatür kalibrasyonu)

Faster R-CNN tabanlı mamografi dedektörleri literatürde genellikle YOLO
ailesiyle benzer büyüklük mertebesinde mAP50 değerleri elde ediyor; ancak
daha büyük girdi çözünürlüğü (800px) küçük kalsifikasyonlar için YOLOv8'e
(640px) göre potansiyel bir avantaj sağlayabilir — bu, raw baseline
karşılaştırmasında gözlemlenecek bir hipotez olarak not edilecek (Calcification
mAP50, YOLOv8'in 0.081 değerinden daha yüksek mi çıkıyor?).

## Sonraki adım

Bu script çalıştırıldıktan sonra: RetinaNet (torchvision, ResNet50-FPN,
COCO-pretrained) raw baseline scripti — aynı `CocoDetectionDataset`
(hazır COCO JSON etiketleri), aynı eğitim/erken durdurma/değerlendirme
iskeleti yeniden kullanılacak, sadece model tanımı değişecek.
