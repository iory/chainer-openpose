import random

import numpy as np


def random_domain_randomize_background(image_rgb, image_mask):
    """

    Ranomly call domain_randomize_background

    """
    if random.random() < 0.5:
        return image_rgb
    else:
        return domain_randomize_background(image_rgb, image_mask)


def domain_randomize_background(image_rgb, image_mask):
    """

    This function applies domain randomization
    to the non-masked part of the image.

    :param image_rgb: rgb image for which the
           non-masked parts of the image will be domain randomized
    :type  image_rgb: PIL.image.image

    :param image_mask: mask of part of image
           to be left alone, all else will be domain randomized
    :type image_mask: PIL.image.image

    :return domain_randomized_image_rgb:
    :rtype: PIL.image.image

    """

    # First, mask the rgb image
    rgb_mask = np.zeros_like(image_rgb)
    rgb_mask[0, :, :] = rgb_mask[1, :, :] = rgb_mask[2, :, :] = \
        image_mask
    image_rgb = image_rgb * rgb_mask

    # Next, domain randomize all non-masked parts of image
    rgb_mask_complement = np.ones_like(rgb_mask) - rgb_mask
    random_rgb_image = get_random_image(image_rgb.shape)
    random_rgb_background = rgb_mask_complement * random_rgb_image

    domain_randomized_image_rgb = image_rgb + random_rgb_background
    return domain_randomized_image_rgb


def get_random_image(shape):
    """
    Expects something like shape=(480,640,3)

    :param shape: tuple of shape for numpy array,
           for example from my_array.shape
    :type shape: tuple of ints

    :return random_image:
    :rtype: np.ndarray
    """
    if random.random() < 0.5:
        rand_image = get_random_solid_color_image(shape)
    else:
        rgb1 = get_random_solid_color_image(shape)
        rgb2 = get_random_solid_color_image(shape)
        vertical = bool(np.random.uniform() > 0.5)
        rand_image = gradient_image(rgb1, rgb2, vertical=vertical)

    if random.random() < 0.5:
        return rand_image
    else:
        return add_noise(rand_image)


def get_random_rgb():
    """

    :return random rgb colors,
     each in range 0 to 255, for example [13, 25, 255]
    :rtype: numpy array with dtype=np.uint8

    """
    return np.array(np.random.uniform(size=3) * 255, dtype=np.uint8)


def get_random_solid_color_image(shape):
    """
    Expects something like shape=(480,640,3)

    :return random solid color image:
    :rtype: numpy array of specificed shape, with dtype=np.uint8
    """
    return np.ones(shape, dtype=np.uint8) * get_random_rgb().reshape(-1, 1, 1)


def get_random_entire_image(shape, max_pixel_uint8):
    """
    Expects something like shape=(480,640,3)

    Returns an array of that shape, with values in range [0..max_pixel_uint8)

    :param max_pixel_uint8: maximum value in the image
    :type max_pixel_uint8: int

    :return random solid color image:
    :rtype: numpy array of specificed shape, with dtype=np.uint8
    """
    return np.array(np.random.uniform(size=shape) *
                    max_pixel_uint8, dtype=np.uint8)


def add_noise(rgb_image):
    """
    Adds noise, and subtracts noise to the rgb_image

    :param rgb_image: image to which noise will be added
    :type rgb_image: numpy array of shape (H,W,3)

    :return image with noise:
    :rtype: same as rgb_image

    ## Note: do not need to clamp, since uint8 will just overflow -- not bad
    """
    max_noise_to_add_or_subtract = 50
    return rgb_image \
        + get_random_entire_image(rgb_image.shape,
                                  max_noise_to_add_or_subtract) \
        - get_random_entire_image(rgb_image.shape,
                                  max_noise_to_add_or_subtract)


def gradient_image(rgb1, rgb2, vertical=False):
    """
    Interpolates between two images rgb1 and rgb2

    :param rgb1, rgb2: two numpy arrays of shape (c,H,W)

    :return interpolated image:
    :rtype: same as rgb1 and rgb2
    """
    bitmap = np.zeros_like(rgb1)
    _, h, w = rgb1.shape
    if vertical:
        p = np.tile(np.linspace(0, 1, h)[:, None], (1, w))
    else:
        p = np.tile(np.linspace(0, 1, w), (h, 1))

    for i in range(3):
        bitmap[i, :, :] = rgb2[i, :, :] * p + rgb1[i, :, :] * (1.0 - p)

    return bitmap
