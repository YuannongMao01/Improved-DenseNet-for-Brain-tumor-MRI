import os
import sys
import copy
import math
import numpy as np
import pandas as pd
import cv2
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from collections import OrderedDict
from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.utils.data import DataLoader,random_split, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score

from brain_tumor_dataset import BrainTumorDataset
from train import train_model
from evaluate import evaluate_model
from Dilated_SEDenseNet_model import SE_DenseNet

print(torch.__version__)

## seed Setting
def set_seed(seed_value):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)  # NumPy
    random.seed(seed_value)  # Python
    torch.manual_seed(seed_value)  # PyTorch
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Python hash build-in
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)


# param setting
lr = 0.003
batch_size = 256
num_workers = 48

dataset_path = './brain-tumor-mri-dataset'
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']

def crop_black_borders(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.threshold
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
#         cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        crop = image[y:y+h, x:x+w]
        return crop
    else:
        return image

def crop_images_in_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for subdir, dirs, files in os.walk(input_directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            cropped_image = crop_black_borders(filepath)
            relative_path = os.path.relpath(filepath, input_directory)
            output_path = os.path.join(output_directory, relative_path)
            output_subdir = os.path.dirname(output_path)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            cv2.imwrite(output_path, cropped_image)

train_dataset_path = f'{dataset_path}/Training'
test_dataset_path = f'{dataset_path}/Testing'

cropped_train_dataset_path = './Training'
cropped_test_dataset_path = './Testing'

crop_images_in_directory(train_dataset_path, cropped_train_dataset_path)
crop_images_in_directory(test_dataset_path, cropped_test_dataset_path)


def show_random_crops(input_directory, num_images=3):
    all_images = []
    for subdir, dirs, files in os.walk(input_directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            all_images.append(filepath)

    selected_images = random.sample(all_images, num_images)

    for i, image_path in enumerate(selected_images):
        image = cv2.imread(image_path)
        cropped_image = crop_black_borders(image_path)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.title('Cropped Image')
        plt.axis('off')
        plt.show()

show_dataset_path = f'{train_dataset_path}/notumor'
show_random_crops(show_dataset_path)

# standard transform
train_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]) 
])

# Create the dataset via MedAugment
train_dataset = BrainTumorDataset(dataset_path=cropped_train_dataset_path,
                                  categories=categories,
                                  transform=train_transform,
                                  med_augment=True,
                                  med_augment_level=5,
                                  med_augment_branch=4)


# create train dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


# Data visualization
def visualize_sample(dataset):
    # Get a random sample
    idx = random.randint(0, len(dataset) - 1)
    img, label = dataset[idx]
    # MRI images are grayscale
    img = transforms.ToPILImage()(img)
    plt.imshow(img, cmap='gray')
    plt.title(dataset.categories[label])
    plt.show()

visualize_sample(train_dataset)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Transforms for the test set
# test_transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize the image to 224x224
#     transforms.Grayscale(num_output_channels=3),
#     transforms.ToTensor(),           # Convert the image to PyTorch tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),  # Normalization
# ])

test_transform = transforms.Compose([
    transforms.Resize(256),  # Resize for TenCrop
    transforms.TenCrop(224),  # generating 10 crops
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),  # First convert to tensor
    # transforms.Lambda(lambda crops: torch.stack([transforms.Grayscale(num_output_channels=3)(crop) for crop in crops])),  # Then apply Grayscale
    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485], std=[0.229])(crop) for crop in crops])),
    # transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops])),  # Finally, normalize
])



test_date = cropped_test_dataset_path

test_dataset = BrainTumorDataset(dataset_path=test_date, categories=categories, transform=test_transform)

# DataLoader for test
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


 # define CrossEntropyLoss
criterion = LabelSmoothingCrossEntropy(4, smoothing=0.2)

from torchvision.models import resnet50, ResNet50_Weights

# New weights with accuracy 80.858%
model_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

for param in model_resnet.parameters():
    param.requires_grad = False

num_ftrs = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(num_ftrs, 4)

model_resnet = nn.DataParallel(model_resnet)

model_resnet = model_resnet.to(device)

optimizer = optim.Adam(model_resnet.module.fc.parameters(), lr=lr, betas=(0.9, 0.999))

## Training

train_losses_resnet, train_accs_resnet, val_losses_resnet, val_accs_resnet = train_model(model_resnet, train_loader,
                                                                             val_loader, optimizer, criterion, device,
                                                                             epochs=50, patience=5,
                                                                             checkpoint_path='pth/resnet_50_model_checkpoint.pth',
                                                                             use_early_stopping=False)


## Evaluate
all_probs_resnet, all_true_resnet, class_metrics_resnet, overall_f1_resnet, overall_recall_resnet = evaluate_model(model_resnet, test_loader, device, categories)

# all_probs_resnet, all_true_resnet, class_metrics_resnet, overall_f1_resnet, overall_recall_resnet

from torchvision.models import vgg16, VGG16_Weights

model_vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

for param in model_vgg.parameters():
    param.requires_grad = False

num_ftrs = model_vgg.classifier[6].in_features
model_vgg.classifier[6] = nn.Linear(num_ftrs, 4)

model_vgg = nn.DataParallel(model_vgg)

model_vgg = model_vgg.to(device)

# optimizer = optim.Adam(model_vgg.classifier[6].parameters(), lr = 0.001, betas=(0.9, 0.999))
optimizer = optim.Adam(model_vgg.module.classifier[6].parameters(), lr=lr, betas=(0.9, 0.999))

## Training
train_losses_vgg, train_accs_vgg, val_losses_vgg, val_accs_vgg = train_model(model_vgg , train_loader,
                                                                             val_loader, optimizer, criterion, device,
                                                                             epochs=50, patience=5,
                                                                             checkpoint_path='pth/vgg_16_model_checkpoint.pth',
                                                                             use_early_stopping=False)

## Evaluate
class_metrics_vgg, overall_f1_vgg, overall_recall_vgg, all_true_vgg, all_preds_vgg = evaluate_model(model_vgg, test_loader, device, categories)


from torchvision.models import vision_transformer

from torchvision.models import vit_l_16, ViT_L_16_Weights

# Load the pretrained Vision Transformer model
model_vit = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)

# Freeze all the parameters
for param in model_vit.parameters():
    param.requires_grad = False

# Replace the classifier head for your specific task
num_ftrs = model_vit.heads[0].in_features
model_vit.heads[0] = nn.Linear(num_ftrs, 4)


model_vit = nn.DataParallel(model_vit)

model_vit = model_vit.to(device)


# optimizer = optim.Adam(model_vit.heads[0].parameters(), lr = 0.001, betas=(0.9, 0.999))
optimizer = optim.Adam(model_vit.module.heads[0].parameters(), lr=lr, betas=(0.9, 0.999))
## Training
train_losses_vit, train_accs_vit, val_losses_vit, val_accs_vit = train_model(model_vit , train_loader,
                                                                             val_loader, optimizer, criterion, device,
                                                                             epochs=50, patience=5,
                                                                             checkpoint_path='pth/vit_model_checkpoint.pth',
                                                                             use_early_stopping=False)

## Evaluate
class_metrics_vit, overall_f1_vit, overall_recall_vit, all_true_vit, all_preds_vit = evaluate_model(model_vit, test_loader, device, categories)




# torch.load('vit_model_checkpoint.pth')

from torchvision.models import densenet121, DenseNet121_Weights

# Load the pretrained DenseNet121 model
model_densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

# Freeze all the parameters
for param in model_densenet.parameters():
    param.requires_grad = False

num_ftrs = model_densenet.classifier.in_features
model_densenet.classifier = nn.Linear(num_ftrs, 4)

model_densenet = nn.DataParallel(model_densenet)
model_densenet = model_densenet.to(device)

# Define the optimizer to optimize only the classifier head
# optimizer = optim.Adam(model_densenet.classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer = optim.Adam(model_densenet.module.classifier.parameters(), lr=lr, betas=(0.9, 0.999))

# Training
train_losses_densenet, train_accs_densenet, val_losses_densenet, val_accs_densenet = train_model(model_densenet, train_loader,
                                                                                                 val_loader, optimizer, criterion,
                                                                                                 device, epochs=50, patience=5,
                                                                                                 checkpoint_path='pth/densenet_model_checkpoint.pth',
                                                                                                 use_early_stopping=False)

## Evaluate
class_metrics_densenet, overall_f1_densenet, overall_recall_densenet, all_true_densenet, all_preds_densenet = evaluate_model(model_densenet, test_loader, device, categories)




from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights

# Load the pretrained EfficientNet V2 M model
model_efficientnet_v2_m = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)

# Freeze all the parameters
for param in model_efficientnet_v2_m.parameters():
    param.requires_grad = False

num_ftrs = model_efficientnet_v2_m.classifier[1].in_features
model_efficientnet_v2_m.classifier[1] = nn.Linear(num_ftrs, 4)

model_efficientnet_v2_m = nn.DataParallel(model_efficientnet_v2_m)
model_efficientnet_v2_m = model_efficientnet_v2_m.to(device)

# Define the optimizer to optimize only the classifier head
optimizer = optim.Adam(model_efficientnet_v2_m.module.classifier[1].parameters(), lr=lr, betas=(0.9, 0.999))

# Training
train_losses_efficientnet, train_accs_efficientnet, val_losses_efficientnet, val_accs_efficientnet = train_model(model_efficientnet_v2_m, train_loader,
                                                                                                                 val_loader, optimizer, criterion,
                                                                                                                 device, epochs=50, patience=5,
                                                                                                                 checkpoint_path='pth/efficientnet_model_checkpoint.pth',
                                                                                                                 use_early_stopping=False
## Evaluate
class_metrics_efficientnet_v2_m, overall_f1_efficientnet_v2_m, overall_recall_efficientnet_v2_m, all_true_efficientnet_v2_m, all_preds_efficientnet_v2_m = evaluate_model(model_efficientnet_v2_m, test_loader, device, categories)





model_self_SE = SE_DenseNet(growthRate=32, head7x7=True, dropRate=0,
                        increasingRate=1, compressionRate=2, layers=(6, 12, 24, 16),
                        num_classes=4)

model_self_SE = nn.DataParallel(model_self_SE)
model_self_SE = model_self_SE.to(device)

optimizer = AdamW(model_self_SE.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)

# ## Training
train_losses_DenseNetSE, train_accs_DenseNetSE, val_losses_DenseNetSE, val_accs_DenseNetSE = train_model(model_self_SE, train_loader,
                                                                                                         val_loader, optimizer,
                                                                                                         criterion, device, epochs=50,
                                                                                                         patience=5,
                                                                                                         checkpoint_path='pth/Dense_SE_checkpoint.pth',
                                                                                                         use_early_stopping=False)

## Evaluate
class_metrics_SE, overall_f1_SE, overall_recall_SE, all_true_SE, all_preds_SE = evaluate_model(model_self_SE, test_loader, device, categories)




import json

output_dict = {}
for model in ['resnet', 'vgg', 'vit', 'densenet', 'efficientnet_v2_m', 'se']:
    output_dict[model] = {}

# output_dict['resnet']['train_losses'] = train_losses_resnet
# output_dict['resnet']['train_acc'] = [float(t.cpu().numpy()) for t in train_accs_resnet]
# output_dict['resnet']['valid_losses'] = val_losses_resnet
# output_dict['resnet']['valid_acc'] = [float(t.cpu().numpy()) for t in val_accs_resnet]
# output_dict['resnet']['by_category_metric'] = class_metrics_resnet
# output_dict['resnet']['overall_f1'] = overall_f1_resnet
# output_dict['resnet']['overall_recall'] = overall_recall_resnet
# output_dict['resnet']['true'] = [float(num) for num in all_true_resnet]
# output_dict['resnet']['preds'] = [float(num) for num in all_preds_resnet]

# output_dict['vgg']['train_losses'] = train_losses_vgg
# output_dict['vgg']['train_acc'] = [float(t.cpu().numpy()) for t in train_accs_vgg]
# output_dict['vgg']['valid_losses'] = val_losses_vgg
# output_dict['vgg']['valid_acc'] = [float(t.cpu().numpy()) for t in val_accs_vgg]
# output_dict['vgg']['by_category_metric'] = class_metrics_vgg
# output_dict['vgg']['overall_f1'] = overall_f1_vgg
# output_dict['vgg']['overall_recall'] = overall_recall_vgg
# output_dict['vgg']['true'] = [float(num) for num in all_true_vgg]
# output_dict['vgg']['preds'] = [float(num) for num in all_preds_vgg]

# output_dict['vit']['train_losses'] = train_losses_vit
# output_dict['vit']['train_acc'] = [float(t.cpu().numpy()) for t in train_accs_vit]
# output_dict['vit']['valid_losses'] = val_losses_vit
# output_dict['vit']['valid_acc'] = [float(t.cpu().numpy()) for t in val_accs_vit]
# output_dict['vit']['by_category_metric'] = class_metrics_vit
# output_dict['vit']['overall_f1'] = overall_f1_vit
# output_dict['vit']['overall_recall'] = overall_recall_vit
# output_dict['vit']['true'] = [float(num) for num in all_true_vit]
# output_dict['vit']['preds'] = [float(num) for num in all_preds_vit]


# output_dict['densenet']['train_losses'] = train_losses_densenet
# output_dict['densenet']['train_acc'] = [float(t.cpu().numpy()) for t in train_accs_densenet]
# output_dict['densenet']['valid_losses'] = val_losses_densenet
# output_dict['densenet']['valid_acc'] = [float(t.cpu().numpy()) for t in val_accs_densenet]
# output_dict['densenet']['by_category_metric'] = class_metrics_densenet
# output_dict['densenet']['overall_f1'] = overall_f1_densenet
# output_dict['densenet']['overall_recall'] = overall_recall_densenet
# output_dict['densenet']['true'] = [float(num) for num in all_true_densenet]
# output_dict['densenet']['preds'] = [float(num) for num in all_preds_densenet]

# output_dict['efficientnet_v2_m']['train_losses'] = train_losses_efficientnet
# output_dict['efficientnet_v2_m']['train_acc'] = [float(t.cpu().numpy()) for t in train_accs_efficientnet]
# output_dict['efficientnet_v2_m']['valid_losses'] = val_losses_efficientnet
# output_dict['efficientnet_v2_m']['valid_acc'] = [float(t.cpu().numpy()) for t in val_accs_efficientnet]
# output_dict['efficientnet_v2_m']['by_category_metric'] = class_metrics_efficientnet_v2_m
# output_dict['efficientnet_v2_m']['overall_f1'] = overall_f1_efficientnet_v2_m
# output_dict['efficientnet_v2_m']['overall_recall'] = overall_recall_efficientnet_v2_m
# output_dict['efficientnet_v2_m']['true'] = [float(num) for num in all_true_efficientnet_v2_m]
# output_dict['efficientnet_v2_m']['preds'] = [float(num) for num in all_preds_efficientnet_v2_m]


output_dict['se']['train_losses'] = train_losses_DenseNetSE
output_dict['se']['train_acc'] = [float(t.cpu().numpy()) for t in train_accs_DenseNetSE]
output_dict['se']['valid_losses'] = val_losses_DenseNetSE
output_dict['se']['valid_acc'] = [float(t.cpu().numpy()) for t in val_accs_DenseNetSE]
output_dict['se']['by_category_metric'] = class_metrics_SE
output_dict['se']['overall_f1'] = overall_f1_SE
output_dict['se']['overall_recall'] = overall_recall_SE
output_dict['se']['true'] = [float(num) for num in all_true_SE]
output_dict['se']['preds'] = [float(num) for num in all_preds_SE]


with open('final_data.json', 'w') as json_file:
    json.dump(output_dict, json_file)

plt.figure(figsize=(10, 6))
plt.style.use('default')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# plt.plot([x.cpu().numpy() for x in train_accs_resnet], label='ResNet50')
# plt.plot([x.cpu().numpy() for x in train_accs_vgg], label='VGG16')
# plt.plot([x.cpu().numpy() for x in train_accs_vit], label='ViT_L_16')
# plt.plot([x.cpu().numpy() for x in train_accs_densenet], label='DenseNet121')
# plt.plot([x.cpu().numpy() for x in train_accs_efficientnet], label='EfficientNet_V2_M')
plt.plot([x.cpu().numpy() for x in train_accs_DenseNetSE], label='Our_model')

plt.plot([x.cpu().numpy() for x in train_accs_DenseNetSE_2], label='improved_1')

plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.style.use('default')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# plt.plot([x.cpu().numpy() for x in train_accs_resnet], label='ResNet50')
# plt.plot([x.cpu().numpy() for x in train_accs_vgg], label='VGG16')
# plt.plot([x.cpu().numpy() for x in train_accs_vit], label='ViT_L_16')
# plt.plot([x.cpu().numpy() for x in train_accs_densenet], label='DenseNet121')
# plt.plot([x.cpu().numpy() for x in train_accs_efficientnet], label='EfficientNet_V2_M')
plt.plot(val_losses_DenseNetSE, label='Our_model')
plt.plot(val_losses_DenseNetSE_2, label='Our_model')

plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()

