from __future__ import division

import cv2
import numpy as np


def resize(img, ignore_mask, poses, shape):
    """resize img, mask and annotations"""
    height, width, _ = img.shape

    resized_img = cv2.resize(img, shape)
    ignore_mask = cv2.resize(ignore_mask.astype(
        np.uint8), shape).astype('bool')
    poses[:, :, :2] = (poses[:, :, :2] * np.array(shape) /
                       np.array((width, height)))
    return resized_img, ignore_mask, poses
