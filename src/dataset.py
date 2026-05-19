import os
import cv2
import torch

from torch.utils.data import Dataset


class BoneDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted(os.listdir(image_dir))

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image_name = self.images[index]

        base_name = os.path.splitext(image_name)[0]

        mask_name = base_name + ".png"

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        image = image / 255.0
        mask = mask / 255.0

        image = torch.tensor(image).permute(2, 0, 1).float()

        mask = torch.tensor(mask).unsqueeze(0).float()

        return image, mask