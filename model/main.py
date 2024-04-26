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
# import albumentations as A
from collections import OrderedDict
from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.utils.data import DataLoader,random_split, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
from fvcore.nn import FlopCountAnalysis, flop_count_table

from brain_tumor_dataset import BrainTumorDataset
from train import train_model
from evaluate import evaluate_model
from Dilated_SEDenseNet_model import SE_DenseNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
def show_random_crops(input_directory, num_images=2):
    all_images = []
    for subdir, dirs, files in os.walk(input_directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            all_images.append(filepath)
#     num_images = min(num_images, len(all_images))
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

show_dataset_path = f'{train_dataset_path}/glioma'
show_random_crops(show_dataset_path)


import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import numpy as np
from skimage.filters import threshold_otsu
from PIL import Image

# Custom transformation for Otsu's thresholding
class OtsuThreshold:
    def __call__(self, img):
        # Convert image to grayscale if it's not already
        if img.mode != 'L':
            img = img.convert('L')
        img = np.array(img)
        thresh = threshold_otsu(img)
        img = Image.fromarray((img > thresh).astype(np.uint8) * 255, mode='L')
        return img

# Custom transformation for random zoom
class RandomZoom:
    def __init__(self, min_scale=0.8, max_scale=1.2):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img):
        scale = random.uniform(self.min_scale, self.max_scale)
        w, h = img.size
        new_w, new_h = int(scale * w), int(scale * h)
        img = TF.resize(img, (new_h, new_w))
        img = TF.center_crop(img, (h, w))
        return img

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.5),  # Adjust brightness
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
    transforms.RandomRotation(degrees=45),   # Random rotation
    transforms.RandomAffine(degrees=0, shear=10),  # Shear transformation
    RandomZoom(min_scale=0.9, max_scale=1.1),  # Random zoom
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # Gaussian blur
    # OtsuThreshold(),  # Otsu's thresholding
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])



class BrainTumorDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, categories, transform=None):
        self.dataset_path = dataset_path
        self.categories = categories
        self.transform = transform
        self.images = []
        self.labels = []

        # Load the data
        for idx, category in enumerate(categories):
            category_path = os.path.join(dataset_path, category)
            for img_name in os.listdir(category_path):
                self.images.append(os.path.join(category_path, img_name))
                self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        # Load image
        image = Image.open(img_path)
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        return image, label



# Create the dataset
train_dataset = BrainTumorDataset(dataset_path=cropped_train_dataset_path,
                            categories=['glioma', 'meningioma', 'notumor', 'pituitary'],
                            transform=train_transform)

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

test_transform = transforms.Compose([
    transforms.Resize(256),  # Resize for TenCrop
    transforms.TenCrop(224),  # generating 10 crops
    # transforms.Resize(299),  # Resize for TenCrop
    # transforms.TenCrop(299),  # generating 10 crops
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),  # First convert to tensor
    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485], std=[0.229])(crop) for crop in crops])),
])

test_date = cropped_test_dataset_path

test_dataset = BrainTumorDataset(dataset_path=test_date, categories=categories, transform=test_transform)

# DataLoader for test
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# class LabelSmoothingCrossEntropy(nn.Module):
#     def __init__(self, classes, smoothing=0.1, dim=-1):
#         super(LabelSmoothingCrossEntropy, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.cls = classes
#         self.dim = dim

#     def forward(self, pred, target):
#         pred = pred.log_softmax(dim=self.dim)
#         with torch.no_grad():
#             # true_dist = pred.data.clone()
#             true_dist = torch.zeros_like(pred)
#             true_dist.fill_(self.smoothing / (self.cls - 1))
#             true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#         return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
#  # define CrossEntropyLoss
# criterion = LabelSmoothingCrossEntropy(4, smoothing=0.2)

criterion = torch.nn.CrossEntropyLoss()

input_size = (1, 3, 224, 224)
dummy_input = torch.randn(input_size).to(device)


def calculate_erf(layers):
    receptive_field = 1 
    stride_product = 1  

    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            kernel_size = layer.kernel_size[0]
            stride = layer.stride[0]
            dilation = layer.dilation[0]
            padding = layer.padding[0]

            receptive_field += ((kernel_size - 1) * dilation * stride_product) - (2 * padding)
            stride_product *= stride
        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
            stride = layer.stride
            kernel_size = layer.kernel_size
            if stride is not None and kernel_size is not None:
                receptive_field += ((kernel_size - 1) * stride_product) - (2 * padding)
                stride_product *= stride

    return receptive_field



## resnet50
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from torch.optim import AdamW

# transfer learning
# model_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model_resnet = resnet101(weights = ResNet101_Weights.IMAGENET1K_V2)

for param in model_resnet.parameters():
    param.requires_grad = False

num_ftrs = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(num_ftrs, 4)

model_resnet = nn.DataParallel(model_resnet)

model_resnet = model_resnet.to(device)
optimizer = optim.AdamW(model_resnet.module.fc.parameters(), lr=lr, betas=(0.9, 0.999))

## Training

train_losses_resnet, train_accs_resnet, val_losses_resnet, val_accs_resnet = train_model(model_resnet, train_loader,
                                                                             val_loader, optimizer, criterion, device,
                                                                             epochs=50, patience=5,
                                                                             checkpoint_path='resnet_101_model_checkpoint.pth',
                                                                             use_early_stopping=False)


total_params_resnet = sum(p.numel() for p in model_resnet.parameters())

erf_resnet = calculate_erf(list(model_resnet.modules()))

# Perform FLOP counting
flops_resnet = FlopCountAnalysis(model_resnet, dummy_input)
print(flops_resnet.total())


## Evaluate
class_metrics_resnet, overall_f1_resnet, overall_recall_resnet,overall_accuracy_resnet, all_true_resnet, all_preds_resnet, auroc_scores_resnet,test_error_resnet= evaluate_model(model_resnet, test_loader, device, categories)

## VGG
from torchvision.models import vgg19, VGG19_Weights

model_vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
# model_vgg = vgg16(weights=None)


for param in model_vgg.parameters():
    param.requires_grad = False

num_ftrs = model_vgg.classifier[6].in_features
model_vgg.classifier[6] = nn.Linear(num_ftrs, 4)

model_vgg = nn.DataParallel(model_vgg)

model_vgg = model_vgg.to(device)


optimizer = optim.AdamW(model_vgg.module.classifier[6].parameters(), lr=lr, betas=(0.9, 0.999))

## Training
train_losses_vgg, train_accs_vgg, val_losses_vgg, val_accs_vgg = train_model(model_vgg , train_loader,
                                                                             val_loader, optimizer, criterion, device,
                                                                             epochs=50, patience=5,
                                                                             checkpoint_path='vgg_19_model_checkpoint.pth',
                                                                             use_early_stopping=False)

total_params_vgg = sum(p.numel() for p in model_vgg.parameters())
erf_vgg = calculate_erf(list(model_vgg.modules()))
flops_vgg = FlopCountAnalysis(model_vgg, dummy_input)

## Evaluate
class_metrics_vgg, overall_f1_vgg, overall_recall_vgg, overall_accuracy_vgg, all_true_vgg, all_preds_vgg, auroc_scores_vgg,test_error_vgg = evaluate_model(model_vgg, test_loader, device, categories)



from torchvision.models import densenet121, DenseNet121_Weights

# model_densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
model_densenet = densenet121(weights=None)

for param in model_densenet.parameters():
    param.requires_grad = False

num_ftrs = model_densenet.classifier.in_features
model_densenet.classifier = nn.Linear(num_ftrs, 4)

model_densenet = nn.DataParallel(model_densenet)
model_densenet = model_densenet.to(device)

optimizer = optim.AdamW(model_densenet.module.classifier.parameters(), lr=lr, betas=(0.9, 0.999))
total_params_densenet = sum(p.numel() for p in model_densenet.parameters())
erf_densenet = calculate_erf(list(model_densenet.modules()))
flops_densenet = FlopCountAnalysis(model_densenet, dummy_input)
train_losses_densenet, train_accs_densenet, val_losses_densenet, val_accs_densenet = train_model(
    model_densenet, train_loader, val_loader, optimizer, criterion, device,
    epochs=50, patience=5, checkpoint_path='densenet121_None_model_checkpoint.pth', use_early_stopping=False)
# Check if the checkpoint was saved with DataParallel
if 'module.' in list(checkpoint_densenet.keys())[0]:
    # If your current model is not in DataParallel, adjust the checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_densenet.items()}
else:
    # If your current model is in DataParallel, adjust accordingly
    # This is just an example and might not be needed in your case
    new_state_dict = {'module.' + k: v for k, v in checkpoint_densenet.items()}

checkpoint_densenet = torch.load('model_state_dict_3_3_3_3.pth')
checkpoint_densenet.keys()#['module.epoch']


checkpoint_densenet = torch.load('densenet121_None_model_checkpoint.pth')
model_densenet.load_state_dict(new_state_dict['module.model_state_dict'])
class_metrics_densenet, overall_f1_densenet, overall_recall_densenet, overall_accuracy_densenet, all_true_densenet, all_preds_densenet, auroc_scores_densenet,test_error_densenet = evaluate_model(model_densenet, test_loader, device, categories)


from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

model_mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)


for param in model_mobilenet.parameters():
    param.requires_grad = False


num_ftrs = model_mobilenet.classifier[1].in_features
model_mobilenet.classifier[1] = nn.Linear(num_ftrs, 4)

model_mobilenet = nn.DataParallel(model_mobilenet)

model_mobilenet = model_mobilenet.to(device)

optimizer = optim.AdamW(model_mobilenet.module.classifier.parameters(), lr=lr, betas=(0.9, 0.999))

train_losses_mobilenet, train_accs_mobilenet, val_losses_mobilenet, val_accs_mobilenet = train_model(
    model_mobilenet, train_loader, val_loader, optimizer, criterion, device,
    epochs=50, patience=5, checkpoint_path='mobilenet_v2_model_checkpoint.pth', use_early_stopping=False)


total_params_mobilenet  = sum(p.numel() for p in model_mobilenet.parameters())
erf_mobilenet  = calculate_erf(list(model_mobilenet.modules()))

flops_mobilenet = FlopCountAnalysis(model_mobilenet, dummy_input)
class_metrics_mobilenet, overall_f1_mobilenet, overall_recall_mobilenet, overall_accuracy_mobilenet, all_true_mobilenet, all_preds_mobilenet, auroc_scores_mobilenet,test_error_mobilenet = evaluate_model(
    model_mobilenet, test_loader, device, categories)


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

optimizer = optim.AdamW(model_vit.module.heads[0].parameters(), lr=lr, betas=(0.9, 0.999))

## Training
train_losses_vit, train_accs_vit, val_losses_vit, val_accs_vit = train_model(model_vit , train_loader,
                                                                             val_loader, optimizer, criterion, device,
                                                                             epochs=50, patience=5,
                                                                             checkpoint_path='vit_model_checkpoint.pth',
                                                                             use_early_stopping=False)

total_params_vit = sum(p.numel() for p in model_vit.parameters())
erf_vit = calculate_erf(list(model_vit.modules()))
flops_vit = FlopCountAnalysis(model_vit, dummy_input)

checkpoint_vit = torch.load('vit_model_checkpoint.pth')

# Check if the checkpoint was saved with DataParallel
if 'module.' in list(checkpoint_vit.keys())[0]:
    # If your current model is not in DataParallel, adjust the checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_vit.items()}
else:
    # If your current model is in DataParallel, adjust accordingly
    # This is just an example and might not be needed in your case
    new_state_dict = {'module.' + k: v for k, v in checkpoint_vit.items()}



model_vit.load_state_dict(new_state_dict['module.model_state_dict'])
# class_metrics_densenet, overall_f1_densenet, overall_recall_densenet, overall_accuracy_densenet, all_true_densenet, all_preds_densenet, auroc_scores_densenet,test_error_densenet = evaluate_model(model_densenet, test_loader, device, categories)

## Evaluate
class_metrics_vit, overall_f1_vit, overall_recall_vit, overall_accuracy_vit, all_true_vit, all_preds_vit, auroc_scores_vit,test_error_vit = evaluate_model(model_vit, test_loader, device, categories)

from torchvision.models import swin_b, Swin_B_Weights

model_swin = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)

for param in model_swin.parameters():
    param.requires_grad = False

num_ftrs = model_swin.head.in_features
model_swin.head = nn.Linear(num_ftrs, 4)

model_swin = nn.DataParallel(model_swin).to(device)

optimizer = AdamW(model_swin.module.head.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.001)


train_losses_swin, train_accs_swin, val_losses_swin, val_accs_swin = train_model(
    model_swin, train_loader, val_loader, optimizer, criterion, device,
    epochs=50, patience=5, checkpoint_path='swin_model_checkpoint.pth',
    use_early_stopping=False
)

total_params_swin = sum(p.numel() for p in model_swin.parameters())
erf_swin = calculate_erf(list(model_swin.modules()))
flops_swin = FlopCountAnalysis(model_swin, dummy_input)

class_metrics_swin, overall_f1_swin, overall_recall_swin, overall_accuracy_swin, all_true_swin, all_preds_swin, auroc_scores_swin,test_error_swin = evaluate_model(
    model_swin, test_loader, device, categories
)

from torch.optim import AdamW
# from fvcore.nn import FlopCountAnalysis, flop_count_table

result_dict = {}
eval_dict = {}


#Training
models = {}
for dil in [[1,2,7,9]]:
    kernel_sizes = [3,3,3,3]
    model_self_SE = SE_DenseNet(growthRate=32, LK_head=True, dropRate=0,
                        increasingRate=1, compressionRate=2, layers=(6, 12, 24, 16),
                        num_classes=4, kernel_sizes = kernel_sizes, dilation_layers = dil)
    
    total_params = sum(p.numel() for p in model_self_SE.parameters())
    print(f"Total parameters for dilation {dil}: {total_params}")
    
    # input_size = (1, 3, 224, 224)
    # flop_counter = FlopCountAnalysis(model_self_SE, input_size)
    # print(f"FLOPs for kernel sizes {kernel_sizes}: {flop_count_table(flop_counter)}")
    
    erf = calculate_erf(list(model_self_SE.modules()))
    print(f"ERF for dilation sizes {dil}: {erf}")

    # print(help(flops))
    # print(type(flops.total()))


    model_self_SE = nn.DataParallel(model_self_SE)
    model_self_SE = model_self_SE.to(device)
    
        
    flops = FlopCountAnalysis(model_self_SE, dummy_input)
    print(flops.total())

    optimizer = AdamW(model_self_SE.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    # model_self_SE = torch.load('model_state_dict_' + kernel_sizes + '.pth')
    
    train_losses_DenseNetSE, train_accs_DenseNetSE, val_losses_DenseNetSE, val_accs_DenseNetSE = train_model(model_self_SE, train_loader,
                                                                                                             val_loader, optimizer,
                                                                                                              criterion, device, epochs=50,   
                                                                                                              patience=5,
                                                                                                              checkpoint_path='Dense_SE_checkpoint.pth',
                                                                                                              use_early_stopping=False)
    
    
    dil = '_'.join(list(map(str, dil)))                                                                                                          
    models[dil] = model_self_SE
    result_dict[dil] = {}
    result_dict[dil]['FLOPs'] = flops.total()
    result_dict[dil]['Total_Params'] = total_params
    result_dict[dil]['ERF'] = erf
    result_dict[dil]['train_loss'] = train_losses_DenseNetSE
    result_dict[dil]['train_acc'] = train_accs_DenseNetSE
    result_dict[dil]['val_loss'] = val_losses_DenseNetSE
    result_dict[dil]['val_acc'] = val_accs_DenseNetSE
    
    
    # Evaluation
    class_metrics_SE, overall_f1_SE, overall_recall_SE, overall_accuracy_SE, all_true_SE, all_preds_SE, auroc_scores_SE, test_error_SE = evaluate_model(model_self_SE, test_loader, device, categories)
    # class_metrics_SE, overall_f1_SE, overall_recall_SE, overall_accuracy_SE, all_true_SE, all_preds_SE, auroc_scores_SE = evaluate_model(model_self_SE, test_loader, device, categories)
    eval_dict[dil] = {}
    eval_dict[dil]['metrics_SE'] = class_metrics_SE
    eval_dict[dil]['f1_SE'] = overall_f1_SE
    eval_dict[dil]['recall_SE'] = overall_recall_SE
    eval_dict[dil]['accuracy_SE'] = overall_accuracy_SE
    eval_dict[dil]['true_SE'] = all_true_SE
    eval_dict[dil]['preds_SE'] = all_preds_SE
    eval_dict[dil]['auroc_scores_SE'] = auroc_scores_SE
    eval_dict[dil]['test_error_SE'] = test_error_SE
    torch.save(model_self_SE, 'model_state_dict_' + dil + '.pth')
    
    

print(result_dict)
