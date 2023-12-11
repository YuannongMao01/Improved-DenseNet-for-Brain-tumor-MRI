# Dilated SE-DenseNet for Radiographic Brain Tumor Classification

## Overview
This project presents an innovative adaptation of the DenseNet121 architecture, integrating dilated convolution layers with the Squeeze-and-Excitation (SE) networks for enhanced diagnostic accuracy in brain tumor classification through MRI images.

## Data
We trained and evaluated our model using a comprehensive Kaggle brain tumor dataset comprising 7023 images, classified into four categories, including healthy brain. The dataset was augmented and preprocessed for optimal model training.

## Model Architecture
Our model advances upon the traditional DenseNet-121 architecture, integrating dilated convolution in place of some standard convolutional layers and augmenting with an SE mechanism. These innovations enhance the model’s representation learning capabilities.

## Training
We used the AdamW optimizer with a custom Label Smoothing cross-entropy loss function and employed a Cosine Annealing learning rate scheduler. The model was trained over 50 epochs with a batch size of 256.

## Testing
Our evaluation used a 10-crop method, involving resizing each image to 256 × 256 pixels and producing ten distinct crops per image. The final test report averages the results over these crops.

## Results
The model demonstrated superior learning ability, outperforming pre-trained models: ResNet50, VGG16, ViT_16, DenseNet121, and Efficient_V2 in later training epochs and in testing.

## Future Work
Future research will focus on the implementation of advanced image augmentation techniques, integration of multi-scale network architecture, and adaptive dilation convolution rates.
