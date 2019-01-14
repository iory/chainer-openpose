import random

from chainer_openpose.transforms.box import get_pose_bboxes
from chainer_openpose.transforms.resize import resize


def random_resize(img, ignore_mask, poses,
                  min_box_size=64,
                  max_box_size=512,
                  min_scale=0.5,
                  max_scale=2.0):
    h, w, _ = img.shape
    joint_bboxes = get_pose_bboxes(poses)
    bbox_sizes = (
        (joint_bboxes[:, 2:] - joint_bboxes[:, :2] + 1)**2).sum(axis=1)**0.5

    min_scale = min_box_size / bbox_sizes.min()
    max_scale = max_box_size / bbox_sizes.max()

    min_scale = min(max(min_scale, min_scale), 1)
    max_scale = min(max(max_scale, 1), max_scale)

    scale = float((max_scale - min_scale) * random.random() + min_scale)
    shape = (round(w * scale), round(h * scale))

    resized_img, resized_mask, resized_poses = resize(
        img, ignore_mask, poses, shape)
    return resized_img, resized_mask, poses
