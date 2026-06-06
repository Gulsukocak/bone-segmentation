import os
import cv2
import torch
import random


from torch.utils.data import Dataset


class BoneDataset(Dataset):

    def __init__(self, image_dir, mask_dir, augment=False):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment

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

        if self.augment:

            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)

            if random.random() > 0.5:
                angle = random.randint(-10, 10)

                h, w = image.shape[:2]

                M = cv2.getRotationMatrix2D(
                    (w // 2, h // 2),
                    angle,
                    1.0
                )

                image = cv2.warpAffine(
                    image,
                    M,
                    (w, h)
                )

                mask = cv2.warpAffine(
                    mask,
                    M,
                    (w, h)
                )

        image = image / 255.0
        mask = mask / 255.0

        image = torch.tensor(image).permute(2, 0, 1).float()

        mask = torch.tensor(mask).unsqueeze(0).float()

        return image, mask
