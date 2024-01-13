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

