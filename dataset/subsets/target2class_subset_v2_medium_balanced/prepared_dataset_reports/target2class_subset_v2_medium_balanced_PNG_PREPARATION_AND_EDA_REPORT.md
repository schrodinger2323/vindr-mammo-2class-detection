# target2class_subset_v2_medium_balanced PNG Preparation and EDA Report

## Purpose

This report summarizes the DICOM-to-PNG conversion and EDA process for the two-class VinDr-Mammo subset.

Target classes:

- Mass
- Suspicious Calcification

## Annotation Policy

The original target annotation table contains exploded target labels.
For model training, annotations were converted into a primary 2-class detection format.

Primary rule:

- If the same bbox contains both Mass and Suspicious Calcification, the primary label is set to Mass.
- Otherwise, the original target class is used.

## Raw PNG Dataset

Prepared dataset:

`/content/drive/MyDrive/vindr_mammo/dataset/prepared_datasets/target2class_subset_v2_medium_balanced_raw_png_2class`

### Raw Split Summary

| dataset        | split   |   image_count |   positive_images |   negative_images |   positive_image_rate |   empty_label_files |   box_count |   Mass_boxes |   Suspicious_Calcification_boxes |
|:---------------|:--------|--------------:|------------------:|------------------:|----------------------:|--------------------:|------------:|-------------:|---------------------------------:|
| raw_png_2class | train   |          1418 |               486 |               932 |              0.342736 |                 932 |         664 |          346 |                              318 |
| raw_png_2class | val     |           306 |               103 |               203 |              0.336601 |                 203 |         131 |           77 |                               54 |
| raw_png_2class | test    |           308 |                95 |               213 |              0.308442 |                 213 |         130 |           69 |                               61 |

### Raw Primary Class Distribution

| class_name               |   box_count |
|:-------------------------|------------:|
| Mass                     |         492 |
| Suspicious Calcification |         433 |

## Crop + CLAHE Dataset

Prepared dataset:

`Not created`

## Important Files

Raw dataset:
- `/content/drive/MyDrive/vindr_mammo/dataset/prepared_datasets/target2class_subset_v2_medium_balanced_raw_png_2class/data.yaml`
- `/content/drive/MyDrive/vindr_mammo/dataset/prepared_datasets/target2class_subset_v2_medium_balanced_raw_png_2class/annotations`
- `/content/drive/MyDrive/vindr_mammo/dataset/prepared_datasets/target2class_subset_v2_medium_balanced_raw_png_2class/eda`

Crop+CLAHE dataset:
- `/content/drive/MyDrive/vindr_mammo/dataset/prepared_datasets/target2class_subset_v2_medium_balanced_crop_clahe_2class/data.yaml`
- `/content/drive/MyDrive/vindr_mammo/dataset/prepared_datasets/target2class_subset_v2_medium_balanced_crop_clahe_2class/annotations`
- `/content/drive/MyDrive/vindr_mammo/dataset/prepared_datasets/target2class_subset_v2_medium_balanced_crop_clahe_2class/eda`

## Next Steps

Recommended experiments:

1. YOLOv8 raw image 2-class baseline.
2. YOLOv8 crop+CLAHE 2-class baseline.
3. Faster R-CNN on the better preprocessing branch.
4. Confidence/NMS threshold optimization.
5. Patch/tile-based high-resolution experiment, especially for Suspicious Calcification.
