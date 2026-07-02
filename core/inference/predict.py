"""Model-forward helpers for inference."""

import numpy as np

from core.data.datasets import HEIGHT_NORM_CONSTANT


def input_channels(sample_img):
    if isinstance(sample_img, (tuple, list)):
        return sample_img[0].shape[0], sample_img[1].shape[0]
    return sample_img.shape[0]


def batched(img_tensor):
    if isinstance(img_tensor, (tuple, list)):
        return tuple(t.unsqueeze(0) for t in img_tensor)
    return img_tensor.unsqueeze(0)


def predict_batch(model, img_batch):
    return model(img_batch).squeeze(0)


def prediction_to_numpy(pred_tensor, *, thresholds=None):
    """Convert a model output tensor to saved prediction layout."""
    pred = pred_tensor.cpu().numpy().astype(np.float32)
    pred[3] = pred[3] * HEIGHT_NORM_CONSTANT

    if thresholds is not None:
        for c, threshold in enumerate(thresholds):
            pred[c] = (pred[c] > threshold).astype(np.float32)

    return pred
