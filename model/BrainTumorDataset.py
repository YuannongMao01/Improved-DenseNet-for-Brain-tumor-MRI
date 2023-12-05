'''
The following code implement the MedAugment and cited from:
title: MedAugment: Universal Automatic Data Augmentation Plug-in for Medical Image Analysis
author: Zhaoshan Liu and Qiujie Lv and Yifan Li and Ziduo Yang and Lei Shen
year:2023
https://arxiv.org/abs/2306.17466
'''

import albumentations as A
import random
import math

def make_odd(num):
    num = math.ceil(num)
    if num % 2 == 0:
        num += 1
    return num

class BrainTumorDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, categories, transform=None, med_augment=False, 
                         med_augment_level=5, med_augment_branch=4):
        
        self.dataset_path = dataset_path
        self.categories = categories
        self.transform = transform
        self.med_augment = med_augment
        self.images = []
        self.labels = []

        # Load data
        for idx, category in enumerate(categories):
            category_path = os.path.join(dataset_path, category)
            for img_name in os.listdir(category_path):
                self.images.append(os.path.join(category_path, img_name))
                self.labels.append(idx)

        # If MedAugment applied
        self.dynamic_transform = None
        if med_augment:
            self.dynamic_transform = BrainTumorDataset.med_augment_transform(med_augment_level, med_augment_branch)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        # Apply MedAugment
        if self.med_augment and self.dynamic_transform is not None:
            dynamic_transform = self.dynamic_transform()
            augmented = dynamic_transform(image=np.array(image))
            image = Image.fromarray(augmented['image'])

        return image, label


    def med_augment_transform(level=5, number_branch=4):
        # define all the transform
        base_transforms = A.Compose([
            A.ColorJitter(brightness=0.04 * level, contrast=0, saturation=0, hue=0, p=0.2 * level),
            A.ColorJitter(brightness=0, contrast=0.04 * level, saturation=0, hue=0, p=0.2 * level),
            A.Posterize(num_bits=math.floor(8 - 0.8 * level), p=0.2 * level),
            A.Sharpen(alpha=(0.04 * level, 0.1 * level), lightness=(1, 1), p=0.2 * level),
            A.GaussianBlur(blur_limit=(3, make_odd(3 + 0.8 * level)), p=0.2 * level),
            A.GaussNoise(var_limit=(2 * level, 10 * level), mean=0, per_channel=True, p=0.2 * level),
            A.Rotate(limit=4 * level, interpolation=1, border_mode=0, value=0, mask_value=None, rotate_method='largest_box', crop_border=False, p=0.2 * level),
            A.HorizontalFlip(p=0.2 * level),
            A.VerticalFlip(p=0.2 * level),
            A.Affine(scale=(1 - 0.04 * level, 1 + 0.04 * level), translate_percent=None, translate_px=None, rotate=None, shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, keep_ratio=True, p=0.2 * level),
            A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None, shear={'x': (0, 2 * level), 'y': (0, 0)}, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, keep_ratio=True, p=0.2 * level),  # x
            A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None, shear={'x': (0, 0), 'y': (0, 2 * level)}, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, keep_ratio=True, p=0.2 * level),
            A.Affine(scale=None, translate_percent={'x': (0, 0.02 * level), 'y': (0, 0)}, translate_px=None, rotate=None, shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, keep_ratio=True, p=0.2 * level),
            A.Affine(scale=None, translate_percent={'x': (0, 0), 'y': (0, 0.02 * level)}, translate_px=None, rotate=None, shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, keep_ratio=True, p=0.2 * level)
        ])

        def get_dynamic_transforms():
            strategy = [(1, 2), (0, 3), (0, 2), (1, 1)]
            if number_branch != 4:
                employ = random.choice(strategy)
            else:
                index = random.randrange(len(strategy))
                employ = strategy.pop(index)
            level_transforms, shape_transforms = random.sample(base_transforms.transforms[:6], employ[0]), random.sample(base_transforms.transforms[6:], employ[1])
            return A.Compose([*level_transforms, *shape_transforms])

        return get_dynamic_transforms
