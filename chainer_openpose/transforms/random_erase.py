import numpy as np


def random_erasing(img, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3.3,
                   copy=False):
    if copy:
        target_img = img.copy()
    else:
        target_img = img

    if p < np.random.rand():
        return target_img

    H, W, _ = target_img.shape
    S = H * W

    while True:
        Se = np.random.uniform(sl, sh) * S
        re = np.random.uniform(r1, r2)

        He = int(np.sqrt(Se * re))
        We = int(np.sqrt(Se / re))

        xe = np.random.randint(0, W)
        ye = np.random.randint(0, H)

        if xe + We <= W and ye + He <= H:
            break

    mask = np.random.randint(0, 255, (He, We, 3))
    target_img[ye:ye + He, xe:xe + We, :] = mask
    return target_img
