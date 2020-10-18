import numpy as np
import cv2


def overlay_heatmap(img, heatmap):
    rgb_heatmap = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img = cv2.addWeighted(img, 0.6, rgb_heatmap, 0.4, 0)
    return img


def overlay_heatmaps(img, heatmaps):
    return overlay_heatmap(img, heatmaps[:-1].max(axis=0))


def overlay_ignore_mask(img, ignore_mask):
    img = img * np.repeat(
        (ignore_mask == 0).astype(np.uint8)
        [:, :, None], 3, axis=2)
    return img


def overlay_paf(img, paf):
    hue = ((np.arctan2(paf[1], paf[0]) / np.pi) / -2 + 0.5)
    saturation = np.sqrt(paf[0] ** 2 + paf[1] ** 2)
    saturation[saturation > 1.0] = 1.0
    value = saturation.copy()
    hsv_paf = np.vstack((hue[np.newaxis],
                         saturation[np.newaxis],
                         value[np.newaxis])).transpose(1, 2, 0)
    rgb_paf = cv2.cvtColor((hsv_paf * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = cv2.addWeighted(img, 0.6, rgb_paf, 0.4, 0)
    return img


def overlay_pafs(img, pafs):
    mix_paf = np.zeros((2,) + img.shape[:-1])
    paf_flags = np.zeros(mix_paf.shape)  # for constant paf

    for paf in pafs.reshape((int(pafs.shape[0]/2), 2,) + pafs.shape[1:]):
        paf_flags = paf != 0
        paf_flags += np.broadcast_to(paf_flags[0] | paf_flags[1], paf.shape)
        mix_paf += paf

    mix_paf[paf_flags > 0] /= paf_flags[paf_flags > 0]
    img = overlay_paf(img, mix_paf)
    return img


def overlay_pose(img, poses, joint_index_pairs,
                 skip_connection_indices=None):
    if skip_connection_indices is None:
        skip_connection_indices = []
    if len(poses) == 0:
        return img

    connection_colors = [
        [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255],
        [0, 85, 255], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0.],
        [255, 0, 85], [170, 255, 0], [85, 255, 0], [170, 0, 255.], [0, 0, 255],
        [0, 0, 255], [255, 0, 255], [170, 0, 255], [255, 0, 170],
    ]

    keypoint_colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
        [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
        [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
        [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    canvas = img.copy()

    # limbs
    for pose in poses.round().astype('i'):
        for i, (limb, color) in enumerate(
                zip(joint_index_pairs, connection_colors)):
            if i not in skip_connection_indices:
                limb_ind = np.array(limb)
                if np.all(pose[limb_ind][:, 2] != 0):
                    joint1, joint2 = pose[limb_ind][:, :2]
                    cv2.line(canvas, tuple(joint1), tuple(joint2), color, 2)

    # joints
    for pose in poses.round().astype('i'):
        for i, ((x, y, v), color) in enumerate(zip(pose, keypoint_colors)):
            if v != 0:
                cv2.circle(canvas, (x, y), 3, color, -1)
    return canvas


def overlay_keypoints(img, keypoints,
                      radius=3, color=(0, 0, 255)):
    """

    keypoints :
        (n, number_of_point in instance, 3)

    """
    img = img.copy()
    for kp in keypoints:
        for x, y, _ in kp:
            cv2.circle(
                img,
                (int(x), int(y)),
                radius,
                color,
                -1)
    return img
