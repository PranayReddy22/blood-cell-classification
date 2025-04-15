# 🧬 Blood Cell Classification using CNNs and YOLOv8

This project explores the use of deep learning for automating hematological analysis. It focuses on identifying and classifying three primary blood cell types—White Blood Cells (WBC), Red Blood Cells (RBC), and Platelets—and further classifies WBCs into four subtypes using convolutional neural networks (CNNs).

---

## 🔬 Objectives

- **Object Detection**  
  Use YOLOv8 to detect and classify WBC, RBC, and Platelets in stained blood smear images.

- **WBC Subtype Classification**  
  Apply ResNet-50 to classify WBCs into:
  - Eosinophils
  - Lymphocytes
  - Monocytes
  - Neutrophils

---

## 🧪 Datasets

- [BCCD Dataset (for object detection)](https://github.com/Shenggan/BCCD_Dataset)
- [Blood Cell Images Dataset (for subtype classification)](https://www.kaggle.com/datasets/paultimothymooney/blood)

---

## 🧠 Methodology

- **Object Detection**:  
  YOLOv8 with:
  - Adaptive Histogram Equalization
  - Data Augmentation (rotation, flipping)
  
- **Subtype Classification**:  
  ResNet-50 via Transfer Learning, compared against:
  - VGG
  - MobileNet
  - SVM
  - Random Forest
  - KNN
  - XGBoost

---

## 📈 Results

### 🔍 Object Detection (YOLOv8)
- **mAP (Mean Average Precision)**: 0.93
- **WBC mAP**: 0.983

### 🧬 WBC Subtype Classification (ResNet-50)
- **Accuracy**: 87.6%
- Best performance among all tested models

---

## 🛠️ Technologies Used

- Python
- YOLOv8 (Ultralytics)
- TensorFlow / Keras
- OpenCV
- Roboflow
- Matplotlib
- Scikit-learn

---
