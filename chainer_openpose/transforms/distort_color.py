import cv2
import numpy as np


def distort_color(img):
    img_max = np.broadcast_to(np.array(255, dtype=np.uint8), img.shape[:-1])
    img_min = np.zeros(img.shape[:-1], dtype=np.uint8)

    hsv_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV).astype(np.int32)
    # hue
    hsv_img[:, :, 0] = np.maximum(
        np.minimum(hsv_img[:, :, 0] - 10 + np.random.randint(20 + 1), img_max),
        img_min)
    # saturation
    hsv_img[:, :, 1] = np.maximum(np.minimum(
        hsv_img[:, :, 1] - 40 + np.random.randint(80 + 1), img_max), img_min)
    # value
    hsv_img[:, :, 2] = np.maximum(np.minimum(
        hsv_img[:, :, 2] - 30 + np.random.randint(60 + 1), img_max), img_min)
    hsv_img = hsv_img.astype(np.uint8)

    distorted_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return distorted_img
