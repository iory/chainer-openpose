import cv2
import numpy as np

from chainer_openpose.datasets.coco.coco_utils import COCO2014JointType


def swap_joints(poses, joint_type_1, joint_type_2):
    tmp = poses[:, joint_type_1].copy()
    poses[:, joint_type_1] = poses[:, joint_type_2]
    poses[:, joint_type_2] = tmp


def flip(img, mask, poses):
    flipped_img = cv2.flip(img, 1)
    flipped_mask = cv2.flip(mask.astype(np.uint8), 1).astype('bool')
    poses[:, :, 0] = img.shape[1] - 1 - poses[:, :, 0]

    swap_joints(poses, COCO2014JointType.LeftEye,
                COCO2014JointType.RightEye)
    swap_joints(poses, COCO2014JointType.LeftEar,
                COCO2014JointType.RightEar)
    swap_joints(poses, COCO2014JointType.LeftShoulder,
                COCO2014JointType.RightShoulder)
    swap_joints(poses, COCO2014JointType.LeftElbow,
                COCO2014JointType.RightElbow)
    swap_joints(poses, COCO2014JointType.LeftHand,
                COCO2014JointType.RightHand)
    swap_joints(poses, COCO2014JointType.LeftWaist,
                COCO2014JointType.RightWaist)
    swap_joints(poses, COCO2014JointType.LeftKnee,
                COCO2014JointType.RightKnee)
    swap_joints(poses, COCO2014JointType.LeftFoot,
                COCO2014JointType.RightFoot)
    return flipped_img, flipped_mask, poses
