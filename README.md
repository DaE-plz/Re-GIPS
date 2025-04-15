# Novel-view X-ray Projection Synthesis via Geometry-Integrated Deep Learning

This repository contains the implementation and enhancement of the **DL-GIPS** architecture for synthesizing novel-view X-ray projections from 3D CT data.  
The project was carried out from **August 2023 to April 2024**.

---

## 🧠 Project Description

The goal of this project is to reproduce and extend the **Geometry-Integrated Projection Synthesis (DL-GIPS)** method using deep learning.  
It focuses on generating accurate X-ray projections at unseen view angles, based on existing CT volumes and known geometrical conditions.

---

## 🔍 Key Features

- Integration of **cone-beam geometry** with **deep neural networks**
- Joint extraction of **texture and geometric features** from lung CT scans
- Use of **generative adversarial networks (GANs)** for realistic projection synthesis
- Multi-stream neural architecture combining several deep networks in parallel
- Training and evaluation on the **LIDC-IDRI** lung CT dataset

---

## 🧪 Methods & Architecture

- **Input**: Lung CT scans and known projection angles
- **Output**: Synthetic X-ray projections at novel angles
- **Backbone**: Modified DL-GIPS architecture combining:
  - Feature encoders (for geometry and appearance)
  - Viewpoint transformation modules
  - GAN-based projection synthesis blocks
- **Loss functions**:
  - Adversarial loss (GAN)
  - Reconstruction loss
  - Consistency loss

---

## 📊 Evaluation

Compared with baseline architectures (e.g., UNet), the improved DL-GIPS achieves:

- ✅ Higher **SSIM** (Structural Similarity Index)
- ✅ Higher **pASNR** (Perceptual Peak Signal-to-Noise Ratio)

---

## 📦 Dataset

- **LIDC-IDRI**: A publicly available lung CT dataset with diverse anatomical and pathological patterns  
- Preprocessing includes CT normalization, resampling, and synthetic X-ray generation

---

## 🧩 Applications

- Viewpoint augmentation for digital radiography
- Simulation-based training in interventional planning
- Improving robustness of computer-aided diagnosis (CADx) systems

---

## 📝 Acknowledgements

This work builds on and extends the original DL-GIPS framework.  
All synthetic X-ray views are non-diagnostic and generated for research purposes only.

---

Maintained by **Daiqi Liu** (daiqi.deutschfau.liu@fau.de)
Affiliation: FAU Erlangen-Nürnberg
