# VinDr-Mammo 2-Class Target Subset Final Download Report

Subset
Subset name: target2class_subset_v2_medium_balanced
Target classes:
Mass
Suspicious Calcification
Selection Summary
Selected studies: 510
Selected images / expected DICOM files: 2040
Manifest path: /content/drive/MyDrive/vindr_mammo/dataset/subsets/target2class_subset_v2_medium_balanced/subset_manifest.csv
URL list: /content/drive/MyDrive/vindr_mammo/dataset/subsets/target2class_subset_v2_medium_balanced/selected_dicom_urls.txt
Split Summary

| subset_split   |   study_count |   image_count |   positive_target_images |   mass_positive_images |   calc_positive_images |   negative_images |   positive_image_rate |
|:---------------|--------------:|--------------:|-------------------------:|-----------------------:|-----------------------:|------------------:|----------------------:|
| test           |            77 |           308 |                       95 |                     59 |                     59 |               213 |                0.3084 |
| train          |           356 |          1424 |                      487 |                    311 |                    299 |               937 |                0.342  |
| val            |            77 |           308 |                      104 |                     69 |                     63 |               204 |                0.3377 |

Target Class Distribution

| class_name               |   bbox_label_count |
|:-------------------------|-------------------:|
| Mass                     |                494 |
| Suspicious Calcification |                519 |

Target Class Distribution by Split

| category_list            |   train |   val |   test |   total |
|:-------------------------|--------:|------:|-------:|--------:|
| Mass                     |     347 |    78 |     69 |     494 |
| Suspicious Calcification |     374 |    73 |     72 |     519 |

Download Summary Before
{
  "expected_dicom_count": 2040,
  "already_available_local_or_drive": 1622,
  "missing_or_invalid": 418,
  "drive_available_count": 1622,
  "local_available_count": 0,
  "drive_available_size_gb": 28.851,
  "local_available_size_gb": 0.0
}
Download Summary After
{
  "expected_dicom_count": 2040,
  "available_local_or_drive": 2040,
  "available_in_drive": 2040,
  "available_in_local": 418,
  "missing_after": 0,
  "drive_available_size_gb": 34.678,
  "local_available_size_gb": 5.827
}
Important Paths
Subset directory: /content/drive/MyDrive/vindr_mammo/dataset/subsets/target2class_subset_v2_medium_balanced
Audit directory: /content/drive/MyDrive/vindr_mammo/dataset/subsets/target2class_subset_v2_medium_balanced/subset_audit
Tables: /content/drive/MyDrive/vindr_mammo/dataset/subsets/target2class_subset_v2_medium_balanced/subset_audit/tables
Figures: /content/drive/MyDrive/vindr_mammo/dataset/subsets/target2class_subset_v2_medium_balanced/subset_audit/figures
Reports: /content/drive/MyDrive/vindr_mammo/dataset/subsets/target2class_subset_v2_medium_balanced/subset_audit/reports
Local raw DICOM directory: /content/vindr_mammo_fast_raw/target2class_subset_v2_medium_balanced
Drive raw DICOM directory: /content/drive/MyDrive/vindr_mammo/dataset/raw/target2class_subset_v2_medium_balanced
Notes

This subset was created for a focused two-class object detection study. It is suitable for the following next steps:

Raw image 2-class YOLOv8 baseline.
Crop + CLAHE 2-class YOLOv8 baseline.
RetinaNet and Faster R-CNN on the same 2-class data.
Confidence/NMS threshold optimization.
High-resolution or patch/tile-based training experiments.
