import os.path as osp

import numpy as np
import cv2
from chainer.dataset import DatasetMixin

from chainer_openpose.datasets.fashion_landmark import fashion_landmark_utils


class FashionLandmarkKeypointsDataset(DatasetMixin):

    def __init__(self, split='train', cloth_type=1,
                 min_keypoints=5):
        super(FashionLandmarkKeypointsDataset, self).__init__()
        self.min_keypoints = min_keypoints
        self.split = split
        if cloth_type != 1:
            raise ValueError()
        self.cloth_type = cloth_type
        self.data_dir = fashion_landmark_utils.get_fashion_landmark()
        self._read_text()

    def _read_text(self):
        eval_list_partition_filename = osp.join(
            self.data_dir,
            'Eval',
            'list_eval_partition.txt')
        landmark_filename = osp.join(
            self.data_dir,
            'Anno',
            'list_landmarks.txt')
        with open(landmark_filename, 'r') as f, \
                open(eval_list_partition_filename, 'r') as eval_list_f:
            eval_list_f.readline()
            eval_list_f.readline()
            f.readline()
            f.readline()
            image_names = []
            clothes_types = []
            variation_types = []
            keypoints = []
            for line, eval_list_line in zip(
                    f.readlines(),
                    eval_list_f.readlines()):
                eval_split = eval_list_line.split()
                split = line.split()
                if split[0] != eval_split[0]:
                    raise ValueError

                if self.split == 'train':
                    if eval_split[1] != 'train':
                        continue
                else:
                    if eval_split[1] == 'train':
                        continue

                cloth_type = int(split[1])
                if cloth_type != self.cloth_type:
                    continue

                if cloth_type == 1:  # upper body
                    if int(len(split[3:]) / 3) != 6:
                        raise ValueError
                    keypoint_num = 6
                elif cloth_type == 2:
                    if int(len(split[3:]) / 3) != 4:
                        raise ValueError
                    keypoint_num = 4
                elif cloth_type == 3:
                    if int(len(split[3:]) / 3) != 8:
                        raise ValueError
                    keypoint_num = 8
                else:
                    raise ValueError('invalid cloth_type')
                keypoint = np.zeros((1, keypoint_num, 3), 'f')

                for i, point_index in enumerate(range(3, len(split), 3)):
                    if int(split[point_index]) == 0:
                        visible = 1.0
                    else:
                        visible = 0.0
                    keypoint[0, i] = np.array(
                        [split[point_index + 1],
                         split[point_index + 2],
                         visible], 'f')
                if np.sum(keypoint[0, :, 2] > 0.0) < self.min_keypoints:
                    continue
                image_names.append(split[0])
                clothes_types.append(cloth_type)
                variation_types.append(int(split[2]))
                keypoints.append(keypoint)
        self.image_names = image_names
        self.keypoints = keypoints
        self.variation_types = variation_types

    def __len__(self):
        return len(self.image_names)

    def get_example(self, i):
        img_filename = osp.join(self.data_dir, self.image_names[i])
        img = cv2.imread(img_filename)
        keypoints = self.keypoints[i]
        ignore_mask = np.zeros((img.shape[0], img.shape[1]), 'f')
        return img, keypoints, ignore_mask


if __name__ == '__main__':
    from chainer_openpose.visualizations import overlay_pafs
    from chainer_openpose.visualizations import overlay_heatmaps
    from chainer_openpose.visualizations import overlay_keypoints
    from chainer_openpose.transforms.heatmap import generate_heatmaps
    from chainer_openpose.transforms.paf import generate_pafs
    from chainer_openpose.datasets.fashion_landmark import \
        fashion_landmark_utils as utils
    dataset = FashionLandmarkKeypointsDataset(split='train')

    index = 0
    while index < len(dataset):
        img, keypoints, _ = dataset[index]
        heatmaps = generate_heatmaps(img, keypoints, sigma=10)
        pafs = generate_pafs(
            img, keypoints, utils.upper_cloth_joint_pairs, sigma=1)
        img = overlay_heatmaps(img, heatmaps)
        img = overlay_pafs(img, pafs)
        img = overlay_keypoints(img, keypoints)
        cv2.imshow(
            'fashion_landmark',
            np.array(img, dtype=np.uint8))

        k = cv2.waitKey(0)
        if k == ord('q'):
            break
        elif k == ord('n'):
            if index == len(dataset) - 1:
                print('WARNING: reached edge index of dataset: %d' % index)
                continue
            index += 1
        elif k == ord('p'):
            if index == 0:
                print('WARNING: reached edge index of dataset: %d' % index)
                continue
            index -= 1
    cv2.destroyAllWindows()
