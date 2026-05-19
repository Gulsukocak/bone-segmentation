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

image, mask = dataset[0]

input_image = image.unsqueeze(0).to(device)

model = UNet().to(device)

model.load_state_dict(
    torch.load("unet_val.pth", map_location=device)
)

model.eval()

with torch.no_grad():

    prediction = model(input_image)

prediction = torch.sigmoid(prediction)

image = image.permute(1, 2, 0).numpy()

mask = mask.squeeze().numpy()

prediction = prediction.squeeze().cpu().numpy()

prediction = (prediction > 0.5).astype(np.float32)

hd = hausdorff_distance(prediction, mask)

print(f"Hausdorff Distance: {hd:.4f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("X-ray")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Ground Truth")

plt.subplot(1, 3, 3)
plt.imshow(prediction, cmap="gray")
plt.title("Prediction")

plt.savefig(
    "../outputs/prediction_result.png",
    bbox_inches="tight"
)

plt.show()