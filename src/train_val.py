import torch
import torch.nn as nn
import random
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from dataset import BoneDataset
import segmentation_models_pytorch as smp
from dice_score import dice_score


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def dice_loss(pred, target, smooth=1e-6):

    pred = torch.sigmoid(pred)

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()

    dice = (2.0 * intersection + smooth) / (
        pred.sum() + target.sum() + smooth
    )

    return 1 - dice


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

dataset = BoneDataset(
    image_dir="../data/images",
    mask_dir="../data/full_masks",
    augment=True
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(device)

bce_loss = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

epochs = 20

for epoch in range(epochs):

    model.train()

    train_loss = 0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        bce = bce_loss(outputs, masks)
        dice = dice_loss(outputs, masks)

        loss = bce + dice

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    model.eval()

    val_loss = 0
    total_dice = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            bce = bce_loss(outputs, masks)
            dice = dice_loss(outputs, masks)

            loss = bce + dice

            val_loss += loss.item()

            dice_metric = dice_score(outputs, masks)

            total_dice += dice_metric

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Dice Score: {total_dice / len(val_loader):.4f}"
    )

torch.save(model.state_dict(), "unet_val.pth")

print("\nTraining completed")
