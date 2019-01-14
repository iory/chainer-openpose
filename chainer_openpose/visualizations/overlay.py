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
