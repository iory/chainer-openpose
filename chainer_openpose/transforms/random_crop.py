import random

import numpy as np

from chainer_openpose.transforms.box import get_pose_bboxes


def random_crop(img, ignore_mask, poses,
                center_perterb_max=40,
                insize=368):
    h, w, _ = img.shape
    joint_bboxes = get_pose_bboxes(poses)
    bbox = random.choice(joint_bboxes)  # select a bbox randomly
    bbox_center = bbox[:2] + (bbox[2:] - bbox[:2])/2

    r_xy = np.random.rand(2)
    perturb = ((r_xy - 0.5) * 2 * center_perterb_max)
    center = (bbox_center + perturb + 0.5).astype('i')

    crop_img = np.zeros((insize, insize, 3), 'uint8') + 127.5
    crop_mask = np.zeros((insize, insize), 'bool')

    offset = (center - (insize-1)/2 + 0.5).astype('i')
    offset_ = (center + (insize-1)/2 - (w-1, h-1) + 0.5).astype('i')

    x1, y1 = (center - (insize-1)/2 + 0.5).astype('i')
    x2, y2 = (center + (insize-1)/2 + 0.5).astype('i')

    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, w-1)
    y2 = min(y2, h-1)

    x_from = -offset[0] if offset[0] < 0 else 0
    y_from = -offset[1] if offset[1] < 0 else 0
    x_to = insize - offset_[0] - 1 if offset_[0] >= 0 else insize - 1
    y_to = insize - offset_[1] - 1 if offset_[1] >= 0 else insize - 1

    crop_img[y_from:y_to+1, x_from:x_to+1] = img[y1:y2+1, x1:x2+1].copy()
    crop_mask[y_from:y_to+1, x_from:x_to +
              1] = ignore_mask[y1:y2+1, x1:x2+1].copy()

    poses[:, :, :2] -= offset
    return crop_img.astype('uint8'), crop_mask, poses
