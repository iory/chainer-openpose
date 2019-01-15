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

    box_min_scale = min_box_size / bbox_sizes.min()
    box_max_scale = max_box_size / bbox_sizes.max()

    box_min_scale = min(max(box_min_scale, min_scale), 1)
    box_max_scale = min(max(box_max_scale, 1), max_scale)

    scale = float((box_max_scale - box_min_scale) *
                  random.random() + box_min_scale)
    shape = (round(w * scale), round(h * scale))

    resized_img, resized_mask, resized_poses = resize(
        img, ignore_mask, poses, shape)
    return resized_img, resized_mask, poses
