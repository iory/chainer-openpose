import numpy as np


def generate_constant_paf(shape, joint_from, joint_to, paf_width):
    """

    Args:
        shape:
             (height, width)
        joint_from:
             (x, y)
        joint_to:
             (x, y)
        paf_width:
              width of paf
    Return:
        constant_paf (`~numpy.ndarray`):
             (2, height, width)

    """
    if np.array_equal(joint_from, joint_to):  # same joint
        return np.zeros((2,) + shape[:-1])

    joint_distance = np.linalg.norm(joint_to - joint_from)
    unit_vector = (joint_to - joint_from) / joint_distance
    rad = np.pi / 2
    rot_matrix = np.array(
        [[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
    vertical_unit_vector = np.dot(rot_matrix, unit_vector)
    grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
    grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
    horizontal_inner_product = unit_vector[0] * (
        grid_x - joint_from[0]) + unit_vector[1] * (grid_y - joint_from[1])
    horizontal_paf_flag = (0 <= horizontal_inner_product) & (
        horizontal_inner_product <= joint_distance)
    vertical_inner_product = vertical_unit_vector[0] * \
        (grid_x - joint_from[0]) + \
        vertical_unit_vector[1] * (grid_y - joint_from[1])
    vertical_paf_flag = np.abs(vertical_inner_product) <= paf_width
    paf_flag = horizontal_paf_flag & vertical_paf_flag
    constant_paf = np.stack((paf_flag, paf_flag)) * \
        np.broadcast_to(unit_vector, shape[:-1] + (2,)).transpose(2, 0, 1)
    return constant_paf


def generate_pafs(img, poses,
                  joint_index_pairs,
                  sigma=8):
    """

    Args:
        img (`~numpy.ndarray`):
             shape of img is ``(height, width, channel)``
        poses (`~numpy.ndarray`):
             shape of poses is ``(number of instances, num_joint, 3)``
        joint_index_pairs:
              [(joint_index from), (joint_index to)] * number of limbs
        sigma (`~float`):
              sigma of gaussian
    Return:
        pafs:
             (number of limbs * 2, height, width)

    """
    _, num_joint, _ = poses.shape
    height, width, _ = img.shape
    n = len(joint_index_pairs)
    pafs = np.zeros((n * 2, height, width), 'f')

    for index, joint_index_pair in enumerate(joint_index_pairs):
        paf = np.zeros((2, height, width), 'f')
        paf_flags = np.zeros((2, height, width), 'f')  # for constant paf

        for pose in poses:
            joint_from, joint_to = pose[list(joint_index_pair)]
            if joint_from[2] > 0 and joint_to[2] > 0:
                limb_paf = generate_constant_paf(
                    img.shape, joint_from[:2], joint_to[:2], sigma)
                limb_paf_flags = limb_paf != 0
                paf_flags += np.broadcast_to(
                    limb_paf_flags[0] | limb_paf_flags[1], limb_paf.shape)
                paf += limb_paf

        paf[paf_flags > 0] /= paf_flags[paf_flags > 0]
        pafs[index * 2:(index + 1) * 2] = paf
    return pafs.astype('f')
