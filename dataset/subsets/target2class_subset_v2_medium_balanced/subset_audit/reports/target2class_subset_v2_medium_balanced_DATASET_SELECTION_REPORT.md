# VinDr-Mammo 2-Class Target Subset Report

## Purpose

This subset was created as if the project had been designed from the beginning as a two-class mammography object detection task.

Target classes:

- Mass
- Suspicious Calcification

## Selection Strategy

The subset includes:

1. All selected studies containing at least one target finding: Mass or Suspicious Calcification.
2. Hard negative studies containing other findings but no target finding.
3. Normal negative studies without target findings.

Study-level splitting was used to avoid data leakage between train, validation, and test sets.

## Paths

- Subset directory: `/content/drive/MyDrive/vindr_mammo/dataset/subsets/target2class_subset_v2_medium_balanced`
- Drive raw DICOM directory: `/content/drive/MyDrive/vindr_mammo/dataset/raw/target2class_subset_v2_medium_balanced`
- Local raw DICOM directory: `/content/vindr_mammo_fast_raw/target2class_subset_v2_medium_balanced`
- Manifest: `/content/drive/MyDrive/vindr_mammo/dataset/subsets/target2class_subset_v2_medium_balanced/subset_manifest.csv`
- URL list: `/content/drive/MyDrive/vindr_mammo/dataset/subsets/target2class_subset_v2_medium_balanced/selected_dicom_urls.txt`

## Configuration

- Target classes: ['Mass', 'Suspicious Calcification']
- Negative study ratio: 0.5
- Hard negative fraction: 0.5
- Max target-positive studies: 340
- Max total selected studies: 520
- Train ratio: 0.7
- Validation ratio: 0.15
- Test ratio: 0.15
- Seed: 42

## Split Summary

| subset_split   |   study_count |   image_count |   positive_target_images |   mass_positive_images |   calc_positive_images |   negative_images |   positive_image_rate |
|:---------------|--------------:|--------------:|-------------------------:|-----------------------:|-----------------------:|------------------:|----------------------:|
| test           |            77 |           308 |                       95 |                     59 |                     59 |               213 |                0.3084 |
| train          |           356 |          1424 |                      487 |                    311 |                    299 |               937 |                0.342  |
| val            |            77 |           308 |                      104 |                     69 |                     63 |               204 |                0.3377 |

## Target Class Distribution

| class_name               |   bbox_label_count |
|:-------------------------|-------------------:|
| Mass                     |                494 |
| Suspicious Calcification |                519 |

## Target Class Distribution by Split

| category_list            |   train |   val |   test |   total |
|:-------------------------|--------:|------:|-------:|--------:|
| Mass                     |     347 |    78 |     69 |     494 |
| Suspicious Calcification |     374 |    73 |     72 |     519 |

## Target BBox Multi-label Summary

|   n_target_classes_for_same_box |   bbox_count |
|--------------------------------:|-------------:|
|                               1 |          841 |
|                               2 |           86 |

## BBox Stats

```json
{
  "n_target_bbox_labels_exploded": 1013,
  "n_unique_target_bboxes": 927,
  "min_bbox_width": 24.219970703200033,
  "median_bbox_width": 221.92993164070003,
  "mean_bbox_width": 267.7791637111676,
  "min_bbox_height": 16.21997070309999,
  "median_bbox_height": 233.64703369138,
  "mean_bbox_height": 286.53221038963244,
  "min_bbox_area": 526.3372573236206,
  "median_bbox_area": 51448.02937747879,
  "mean_bbox_area": 111950.30946493207
}
Notes
This dataset is more appropriate for a focused two-class detection study than the previous 10-class subset.
Non-target findings are not used as detection labels, but studies/images containing non-target findings can be included as hard negatives.
The raw DICOM images are downloaded first to local Colab storage and periodically synchronized to Google Drive.
The dataset can later be converted into raw PNG, crop+CLAHE PNG, and patch/tile-based training sets.
