from torch.utils.data import Dataset
from PIL import Image


class ChestXrayDataset(Dataset):
    def __init__(self, img_paths, transform=None, target_transform=None):
        self.img_paths = img_paths
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = img_path.split("/")[3]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
