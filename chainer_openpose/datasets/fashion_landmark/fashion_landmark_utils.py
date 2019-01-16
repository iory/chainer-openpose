from enum import IntEnum
import os
import shutil

from chainer.dataset import download
from chainercv import utils

from chainer_openpose.datasets.download import cached_gdown_download


root = 'iory/openpose/fashion_landmark'


class UpperClothJointType(IntEnum):

    left_collar = 0
    right_collar = 1
    left_sleeve = 2
    right_sleeve = 3
    left_hem = 4
    right_hem = 5


upper_cloth_joint_pairs = (
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (4, 5))


def get_fashion_landmark():
    data_dir = download.get_dataset_directory(root)
    url = 'https://drive.google.com/uc?id=0B7EVK8r0v71pSU9nOXVDMk9WbWM'
    img_root = os.path.join(data_dir, 'img')
    anno_root = os.path.join(data_dir, 'Anno')
    eval_root = os.path.join(data_dir, 'Eval')
    download_file_path = cached_gdown_download(url)
    if not os.path.exists(img_root):
        utils.extractall(download_file_path, data_dir, '.zip')

    landmark_annotation_url = 'https://drive.google.com/uc?id='\
        '0B7EVK8r0v71pZ3pGVFZ0YjZVTjg'
    download_file_path = cached_gdown_download(landmark_annotation_url)
    try:
        os.makedirs(anno_root)
    except OSError:
        if not os.path.exists(anno_root):
            raise
    shutil.copy(download_file_path,
                os.path.join(anno_root, 'list_landmarks.txt'))

    eval_list_url = 'https://drive.google.com/uc?id='\
        '0B7EVK8r0v71pakJzTEM0a2Q4Qm8'
    download_file_path = cached_gdown_download(eval_list_url)
    try:
        os.makedirs(eval_root)
    except OSError:
        if not os.path.exists(eval_root):
            raise
    shutil.copy(download_file_path,
                os.path.join(eval_root, 'list_eval_partition.txt'))
    return data_dir
