import numpy as np


def generate_gaussian_heatmap(shape, xy, sigma):
    """

    Args:
        shape: (height, width)

    """
    x, y = xy
    grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
    grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
    grid_distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
    gaussian_heatmap = np.exp(-0.5 * grid_distance / sigma**2)
    return gaussian_heatmap


def generate_heatmaps(img, poses, sigma=7.0):
    """

    Args:
        img (`~numpy.ndarray`):
             shape of img is ``(height, width, channel)``
        poses (`~numpy.ndarray`):
             shape of poses is ``(number of instances, num_joint, 3)``
        sigma (`~float`):
              sigma of gaussian

    """
    _, num_joint, _ = poses.shape
    height, width, _ = img.shape
    heatmaps = np.zeros((0, height, width))
    sum_heatmap = np.zeros((height, width))
    for joint_index in range(num_joint):
        heatmap = np.zeros((height, width))
        for pose in poses:
            if pose[joint_index, 2] > 0:
                jointmap = generate_gaussian_heatmap(
                    (height, width), pose[joint_index][:2], sigma)
                heatmap[jointmap > heatmap] = jointmap[jointmap > heatmap]
                sum_heatmap[jointmap >
                            sum_heatmap] = jointmap[jointmap > sum_heatmap]
        heatmaps = np.vstack((heatmaps, heatmap.reshape((1, height, width))))
    bg_heatmap = 1 - sum_heatmap  # background channel
    heatmaps = np.vstack((heatmaps, bg_heatmap[None]))
    return heatmaps.astype('f')
