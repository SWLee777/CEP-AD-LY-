# CEP-AD

## Introduction
The purpose of anomaly detection is to identify and detect abnormal regions in images of types that have never been seen before. Existing research mostly relies on CLIP and other large-scale vision-language models, which have validated the effectiveness of text prompts for recognizing unknown anomalies. However, current methods either adopt fixed text prompts or rely solely on single‑image patch features to update text prompts, and they easily overfit to auxiliary data and struggle to identify truly unknown anomalies. To address this challenge, we propose a new framework called CEP-AD. This framework performs multi-scale patch feature aggregation on images and combines cross-attention mechanisms to capture global image features, thereby addressing the shortcomings of existing research which lacks global representation. By constructing multiple expert pools, relying on image feature pre-optimization expert parameters, and using global feature dynamic generation for expert weights, it achieves dynamic fusion of individual features and global features, effectively purifying feature noise and stabilizing the update of text prompts. In addition, it introduces an anomaly detection calibration strategy that combines multi-scale feature fusion and prompt scoring constraints, calibrating the anomaly detection results at both the image-level and pixel-level to further improve the detection accuracy of the model. Experiments on 10 real datasets covering industrial defects and medical anomalies demonstrate that the performance of CEP-AD is superior to that of current state-of-the-art methods.

## Device
- Single NVIDIA GeForce RTX 4090

## Prepare Your Data
#### Step 1. Download the Anomaly Detection Datasets
- Industrial Anomaly Detection Datasets: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), [VisA](https://github.com/amazon-science/spot-diff), [AITEX](https://www.aitex.es/afid/), [BTAD](http://avires.dimi.uniud.it/papers/btad/btad.zip),  [MPDD](https://github.com/stepanje/MPDD).

- Medical Anomaly Detection Datasets: [HeadCT](https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage), [Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection), [CVC-ColonDB](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579), [CVC-ClinicDB](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579), [Kvasir](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579).

#### Step 2. Generate the JSON file for Datasets (same as [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP/tree/main?tab=readme-ov-file))

#### Step 3. Download the Pre-train Models on [Google Drive](https://drive.google.com/drive/folders/11Z5msKSrnIECamZO4kYy_rciQXS0v9vI?usp=sharing).

## Quick Start
#### Installation
```
conda create -n CEP python=3.8 -y  
conda activate CEP
pip install -r requirements.txt
```

#### Training & Evaluation
- Train your own weights by runing
```bash
bash train.sh
```
- Evaluation of model performance
```bash
bash test.sh
```



