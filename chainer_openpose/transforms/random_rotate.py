import math

import numpy as np
import cv2


def random_rotate(img, mask, poses,
                  max_rotate_degree=40):
    h, w, _ = img.shape
    degree = np.random.randn() / 3 * max_rotate_degree
    rad = degree * math.pi / 180
    center = (w / 2, h / 2)
    R = cv2.getRotationMatrix2D(center, degree, 1)
    bbox = (w*abs(math.cos(rad)) + h*abs(math.sin(rad)),
            w*abs(math.sin(rad)) + h*abs(math.cos(rad)))
    R[0, 2] += bbox[0] / 2 - center[0]
    R[1, 2] += bbox[1] / 2 - center[1]
    rotate_img = cv2.warpAffine(
        img, R,
        (int(bbox[0]+0.5), int(bbox[1]+0.5)),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=[127.5, 127.5, 127.5])
    rotate_mask = cv2.warpAffine(mask.astype(
        'uint8')*255, R, (int(bbox[0]+0.5), int(bbox[1]+0.5))) > 0

    tmp_poses = np.ones_like(poses)
    tmp_poses[:, :, :2] = poses[:, :, :2].copy()

    # apply rotation matrix to the poses
    tmp_rotate_poses = np.dot(tmp_poses, R.T)
    rotate_poses = poses.copy()  # to keep visibility flag
    rotate_poses[:, :, :2] = tmp_rotate_poses
    return rotate_img, rotate_mask, rotate_poses
