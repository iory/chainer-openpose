import cv2
import numpy as np

from chainer_openpose.datasets.coco.coco_utils import JointType


def swap_joints(poses, joint_type_1, joint_type_2):
    tmp = poses[:, joint_type_1].copy()
    poses[:, joint_type_1] = poses[:, joint_type_2]
    poses[:, joint_type_2] = tmp


def flip(img, mask, poses):
    flipped_img = cv2.flip(img, 1)
    flipped_mask = cv2.flip(mask.astype(np.uint8), 1).astype('bool')
    poses[:, :, 0] = img.shape[1] - 1 - poses[:, :, 0]

    swap_joints(poses, JointType.LeftEye,
                JointType.RightEye)
    swap_joints(poses, JointType.LeftEar,
                JointType.RightEar)
    swap_joints(poses, JointType.LeftShoulder,
                JointType.RightShoulder)
    swap_joints(poses, JointType.LeftElbow,
                JointType.RightElbow)
    swap_joints(poses, JointType.LeftHand,
                JointType.RightHand)
    swap_joints(poses, JointType.LeftWaist,
                JointType.RightWaist)
    swap_joints(poses, JointType.LeftKnee,
                JointType.RightKnee)
    swap_joints(poses, JointType.LeftFoot,
                JointType.RightFoot)
    return flipped_img, flipped_mask, poses
