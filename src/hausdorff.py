import numpy as np

from scipy.spatial.distance import directed_hausdorff


def hausdorff_distance(prediction, target):

    prediction = prediction.astype(np.uint8)

    target = target.astype(np.uint8)

    pred_points = np.argwhere(prediction > 0)

    target_points = np.argwhere(target > 0)

    if len(pred_points) == 0 or len(target_points) == 0:
        return None

    hd1 = directed_hausdorff(pred_points, target_points)[0]

    hd2 = directed_hausdorff(target_points, pred_points)[0]

    return max(hd1, hd2)