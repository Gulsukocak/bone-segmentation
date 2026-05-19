import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import BoneDataset
from unet import UNet
from hausdorff import hausdorff_distance


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = BoneDataset(
    image_dir="../data/images",
    mask_dir="../data/full_masks"
)

model = UNet().to(device)

model.load_state_dict(
    torch.load("unet_val.pth", map_location=device)
)

model.eval()

for i in range(5):

    image, mask = dataset[i]

    input_image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        prediction = model(input_image)

    prediction = torch.sigmoid(prediction)

    image_np = image.permute(1, 2, 0).numpy()

    mask_np = mask.squeeze().numpy()

    prediction_np = prediction.squeeze().cpu().numpy()

    prediction_np = (prediction_np > 0.5).astype(np.float32)

    hd = hausdorff_distance(prediction_np, mask_np)

    print(
        f"Image {i+1} | "
        f"Hausdorff Distance: {hd:.4f}"
    )

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("X-ray")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_np, cmap="gray")
    plt.title("Ground Truth")

    plt.subplot(1, 3, 3)
    plt.imshow(prediction_np, cmap="gray")
    plt.title("Prediction")

    plt.savefig(
        f"../outputs/prediction_{i+1}.png",
        bbox_inches="tight"
    )

    plt.close()

print("\nPredictions saved to outputs folder")