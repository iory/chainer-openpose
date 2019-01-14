import numpy as np


def get_pose_bboxes(poses):
    pose_bboxes = []
    for pose in poses:
        x1 = pose[pose[:, 2] > 0][:, 0].min()
        y1 = pose[pose[:, 2] > 0][:, 1].min()
        x2 = pose[pose[:, 2] > 0][:, 0].max()
        y2 = pose[pose[:, 2] > 0][:, 1].max()
        pose_bboxes.append([x1, y1, x2, y2])
    pose_bboxes = np.array(pose_bboxes)
    return pose_bboxes
