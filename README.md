# Hand Tracking and Object Detection for Assembly Stage Verification and Improvement

![Header Image](https://via.placeholder.com/1200x400) <!-- Add a relevant header image if available -->

This project focuses on enhancing assembly process verification through advanced hand tracking and object detection models. It provides robust datasets, optimized models, and comprehensive evaluation metrics to ensure accurate stage verification and efficiency improvements in assembly workflows.

---

## 📖 Project Overview

Our system integrates computer vision techniques to monitor assembly processes in real-time. By leveraging ​**YOLO models** for object detection and ​**custom datasets** for precise action recognition, we enable automated verification of assembly sequences and adherence to standard time thresholds. This repository contains the model training pipeline, dataset construction guidelines, and evaluation tools.

---

## 🚀 Core Features

- ​**Multi-Perspective Dataset**:  
  - ​**Dual-camera Setup**: Main + Side-view videos under controlled lighting & attire.
  - ​**Temporal Action Segmentation**: Keyframe annotations with action phase timestamps.

- ​**Smart Annotation Workflow**:  
  ✅ Roboflow Auto-Labeling Pipeline  
  ✅ Via JSON Batch Annotation Export  
  ✅ Handling/Grasping Action Taxonomy

- ​**Model Training & Evaluation**:  
  🧠 YOLOv8/v9 with MS COCO Pre-Trained Weights  
  📊 Confusion Matrix & F1 Score Vector Analysis  
  ⏱️ Standard Time Compliance Thresholds

- ​**EOID Integration**:  
  🔗 Preset Assembly Sequence Validation  
  🔄 Dataset Migration & Format Conversion Tools

---

## 🔧 Technology Stack

### Dataset Construction
- ​**Annotation Tools**: Roboflow, Via (VGG Image Annotator)
- ​**Preprocessing**: Auto-Orient, EXIF standardization
- ​**Augmentation**:  
  ```yaml
  - Horizontal Flip
  - ±15° Rotation
  - 15% Grayscale Conversion
  - Hue Variation (±18°)
  - Gaussian Blur (σ=2.6px)
