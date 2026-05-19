import torch


def dice_score(prediction, target, smooth=1e-6):

    prediction = torch.sigmoid(prediction)

    prediction = (prediction > 0.5).float()

    prediction = prediction.view(-1)

    target = target.view(-1)

    intersection = (prediction * target).sum()

    dice = (2. * intersection + smooth) / (
        prediction.sum() + target.sum() + smooth
    )

    return dice.item()