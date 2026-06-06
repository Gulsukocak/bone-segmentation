import torch
import torch.nn as nn
import numpy as np
import random

from sklearn.model_selection import KFold

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from dataset import BoneDataset
import segmentation_models_pytorch as smp

from dice_score import dice_score
from hausdorff import hausdorff_distance


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

indices = np.arange(len(dataset))

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):

    print(f"\nFold {fold + 1}")

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

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

    epochs = 10

    for epoch in range(epochs):

        model.train()

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

        model.eval()

        total_dice = 0
        total_hd = 0
        valid_hd_count = 0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)

                dice = dice_score(outputs, masks)

                total_dice += dice

                prediction = torch.sigmoid(outputs)

                prediction = (prediction > 0.5).float()

                pred_np = prediction.squeeze().cpu().numpy()

                mask_np = masks.squeeze().cpu().numpy()

                hd = hausdorff_distance(pred_np, mask_np)

                if hd is not None:

                    total_hd += hd

                    valid_hd_count += 1

        avg_dice = total_dice / len(val_loader)

        avg_hd = total_hd / valid_hd_count

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Dice: {avg_dice:.4f} | "
            f"Hausdorff: {avg_hd:.4f}"
        )

    fold_results.append((avg_dice, avg_hd))

final_dice = np.mean([x[0] for x in fold_results])

final_hd = np.mean([x[1] for x in fold_results])

print("\nFinal Results")

print(f"Average Dice Score: {final_dice:.4f}")

print(f"Average Hausdorff Distance: {final_hd:.4f}")

with open("../reports/cross_validation_results.txt", "w") as file:

    file.write("5-Fold Cross Validation Results\n\n")

    for i, (dice, hd) in enumerate(fold_results):

        file.write(
            f"Fold {i+1} | "
            f"Dice: {dice:.4f} | "
            f"Hausdorff: {hd:.4f}\n"
        )

    file.write("\n")

    file.write(
        f"Average Dice Score: {final_dice:.4f}\n"
    )

    file.write(
        f"Average Hausdorff Distance: {final_hd:.4f}\n"
    )

