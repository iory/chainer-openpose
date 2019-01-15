import numpy as np

from chainer_openpose.transforms.flip import flip


def random_flip(img, mask, poses):
    if np.random.uniform() > 0.5:
        return flip(img, mask, poses)
    else:
        return img, mask, poses
