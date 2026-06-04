"""Model-forward helpers for inference and TTA."""

import torch

from core.data.datasets import HEIGHT_NORM_CONSTANT
from core.data.height_stats import denormalize_height_numpy


def input_channels(sample_img):
    if isinstance(sample_img, (tuple, list)):
        return sample_img[0].shape[0], sample_img[1].shape[0]
    return sample_img.shape[0]


def batched(img_tensor):
    if isinstance(img_tensor, (tuple, list)):
        return tuple(t.unsqueeze(0) for t in img_tensor)
    return img_tensor.unsqueeze(0)


def tta_views(mode):
    if mode == "none":
        return [(0, False)]
    if mode == "flip":
        return [(0, False), (0, True), (2, True)]
    return [(k, mirror) for k in range(4) for mirror in (False, True)]


def transform_tensor(x, rot_k, mirror):
    if rot_k:
        x = torch.rot90(x, rot_k, dims=(-2, -1))
    if mirror:
        x = torch.flip(x, dims=(-1,))
    return x


def invert_tensor(x, rot_k, mirror):
    if mirror:
        x = torch.flip(x, dims=(-1,))
    if rot_k:
        x = torch.rot90(x, -rot_k, dims=(-2, -1))
    return x


def transform_input(img_batch, rot_k, mirror):
    if isinstance(img_batch, (tuple, list)):
        return tuple(transform_tensor(t, rot_k, mirror) for t in img_batch)
    return transform_tensor(img_batch, rot_k, mirror)


def predict_batch(model, img_batch, views):
    if len(views) == 1:
        return model(img_batch).squeeze(0)

    preds = []
    for rot_k, mirror in views:
        aug_input = transform_input(img_batch, rot_k, mirror)
        aug_pred = model(aug_input)
        preds.append(invert_tensor(aug_pred, rot_k, mirror).squeeze(0))
    return torch.stack(preds, dim=0).mean(dim=0)


def prediction_to_numpy(pred_tensor, *, thresholds=None, height_norm_stats=None):
    """Convert a model output tensor to saved prediction layout."""
    pred = pred_tensor.cpu().numpy().astype(np.float32)
    if height_norm_stats is not None:
        pred[3] = denormalize_height_numpy(pred[3], height_norm_stats).astype(np.float32)
    else:
        pred[3] = pred[3] * HEIGHT_NORM_CONSTANT

    if thresholds is not None:
        for c, threshold in enumerate(thresholds):
            pred[c] = (pred[c] > threshold).astype(np.float32)

    return pred
