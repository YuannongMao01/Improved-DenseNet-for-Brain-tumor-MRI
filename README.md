# Dilated SE-DenseNet for Radiographic Brain Tumor Classification

## Overview
This repository implements the enhanced DenseNet121 architecture integrated by dilated convolution and Squeeze-and-Excitation (SE) networks to improve the diagnostic accuracy in brain tumor classification through MRI images.

<img width="782" alt="Screen Shot 2023-12-10 at 10 39 50 PM" src="https://github.com/YuannongMao01/Improved-DenseNet-for-Brain-tumor-MRI/assets/89234579/a6bbfe56-f307-4e04-a74a-3b12ee064747">


## Data
We trained and evaluated our model using a comprehensive Kaggle brain tumor dataset comprising 7023 images, classified into four categories, including healthy brain. The dataset was augmented and preprocessed for optimal model training.

## Model Architecture
Our model advances upon the traditional DenseNet-121 architecture, integrating dilated convolution in place of some standard convolutional layers and augmenting with an SE mechanism. These innovations enhance the model’s representation learning capabilities.

<img width="1102" alt="Screen Shot 2023-12-10 at 10 44 07 PM" src="https://github.com/YuannongMao01/Improved-DenseNet-for-Brain-tumor-MRI/assets/89234579/1e657d3b-dd96-4e39-a2a3-f97662170fa9">

## Training
We used the AdamW optimizer with a custom Label Smoothing cross-entropy loss function and employed a Cosine Annealing learning rate scheduler. The model was trained over 50 epochs with a batch size of 256.

## Testing
Our evaluation used a 10-crop method, involving resizing each image to 256 × 256 pixels and producing ten distinct crops per image. The final test report averages the results over these crops.

## Results
The model demonstrated superior learning ability, outperforming pre-trained models: ResNet50, VGG16, ViT_16, DenseNet121, and Efficient_V2 in later training epochs and in testing.

<img width="1088" alt="Screen Shot 2023-12-10 at 10 49 14 PM" src="https://github.com/YuannongMao01/Improved-DenseNet-for-Brain-tumor-MRI/assets/89234579/ae58fc9d-2886-4d62-8bb7-f7d1f63250a9">


## Future Work
Future research will focus on the implementation of advanced image augmentation techniques, integration of multi-scale network architecture, and adaptive dilation convolution rates.
