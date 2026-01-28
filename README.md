# Dilated SE-DenseNet for Brain Tumor MRI Classification

[![Paper](https://img.shields.io/badge/Paper-Scientific%20Reports-blue)](https://www.nature.com/articles/s41598-025-86752-y)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## Overview

This repository contains the official implementation of our paper published in **Scientific Reports** (Nature Portfolio):

> **[Dilated SE-DenseNet for Radiographic Brain Tumor Classification](https://www.nature.com/articles/s41598-025-86752-y)**

We present an enhanced DenseNet-121 architecture that integrates dilated convolutions and Squeeze-and-Excitation (SE) networks to improve diagnostic accuracy in brain tumor classification from MRI images.

<img width="782" alt="Overview" src="https://github.com/YuannongMao01/Improved-DenseNet-for-Brain-tumor-MRI/assets/89234579/a6bbfe56-f307-4e04-a74a-3b12ee064747">

## Dataset

We trained and evaluated our model using a comprehensive [Kaggle brain tumor dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) comprising 7,023 MRI images classified into four categories:
- Glioma
- Meningioma
- Pituitary tumor
- Healthy brain (no tumor)

The dataset was augmented and preprocessed for optimal model training.

## Model Architecture

Our model builds upon the traditional DenseNet-121 architecture with two key enhancements:
1. **Dilated Convolutions**: Replace standard convolutional layers to expand the receptive field without increasing parameters
2. **Squeeze-and-Excitation (SE) Mechanism**: Enables adaptive channel-wise feature recalibration

These innovations enhance the model's representation learning capabilities for medical image analysis.

<img width="1102" alt="Architecture" src="https://github.com/YuannongMao01/Improved-DenseNet-for-Brain-tumor-MRI/assets/89234579/1e657d3b-dd96-4e39-a2a3-f97662170fa9">

## Training

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Loss Function | Cross-Entropy |
| LR Scheduler | Cosine Annealing |
| Epochs | 50 |
| Batch Size | 256 |

## Evaluation

Our evaluation employs a **10-crop testing method**:
1. Resize each image to 256 × 256 pixels
2. Generate 10 distinct crops per image
3. Average predictions across all crops for the final result

## Results

The Dilated SE-DenseNet demonstrates superior learning ability, outperforming several pre-trained baseline models:
- ResNet-101
- VGG-19
- MobileNet-V2
- ViT-L/16
- Swin-B
- DenseNet-121

<img width="1007" height="521" alt="Results" src="https://github.com/user-attachments/assets/23551434-c966-4d10-8dec-3a3b34973569" />

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{mao2025dilated,
  title={Dilated SE-DenseNet for brain tumor MRI classification},
  author={Mao, Yuannong and Kim, Jiwook and Podina, Lena and Kohandel, Mohammad},
  journal={Scientific Reports},
  volume={15},
  number={1},
  pages={3596},
  year={2025},
  publisher={Nature Publishing Group},
  doi={10.1038/s41598-025-86752-y}
}
```

## Authors

This work was conducted at the University of Waterloo (2024):

- [Yuannong Mao](https://www.linkedin.com/in/yuannongmao) — 4th year undergraduate, Applied Mathematics
- [Jiwook Kim](https://www.linkedin.com/in/edwardjiwookkim) — 4th year undergraduate, Applied Mathematics
- Lena Podina — David R. Cheriton School of Computer Science
- Mohammad Kohandel — Applied Mathematics

## License

This project is licensed under [CC BY-NC-ND 4.0](http://creativecommons.org/licenses/by-nc-nd/4.0/).
