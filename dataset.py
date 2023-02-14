from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import numpy as np


class MDataset(Dataset):
    def __init__(self, data):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.244, 0.225]),
        ])

        self.paths = []
        self.labels = []
        for x, y in data:
            self.paths.append(x)
            self.labels.append(y)

    def __len__(self):
        return len(self.paths)

    def labelcount(self, minlength=0):
        return np.bincount(self.labels, minlength=minlength)

    def undersample(self):
        labelcount = self.labelcount()
        count = [0 for i in labelcount]
        new_paths = []
        new_labels = []
        for i in range(len(self)):
            path = self.paths[i]
            label = self.labels[i]
            if count[label] < labelcount.min():
                new_labels.append(label)
                new_paths.append(path)
                count[label] += 1
        self.paths = new_paths
        self.labels = new_labels

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert('RGB')
        except BaseException:
            img = Image.new('RGB', (256, 256))
        tensor_img = self.transform(img)
        label = self.labels[idx]
        return tensor_img, label, idx
