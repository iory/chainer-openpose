import os
import os.path as osp

from chainer.dataset import DatasetMixin
from pycocotools.coco import COCO
import cv2
import numpy as np

from chainer_openpose.datasets.coco import coco_utils
from chainer_openpose.datasets.coco.coco_utils import JointType


class COCOPersonKeypointsDataset(DatasetMixin):

    def __init__(self, split='train', year='2017',
                 min_keypoints=5,
                 min_area=1024):
        super(COCOPersonKeypointsDataset, self).__init__()
        self.min_keypoints = min_keypoints
        self.min_area = min_area

        self.coco_path = coco_utils.get_coco(split, year)
        self.img_root = os.path.join(
            self.coco_path, "images",
            "{}{}".format(split, year))
        self.annos_root = osp.join(
            self.coco_path,
            'annotations')
        self.coco = COCO(
            osp.join(
                self.annos_root,
                'person_keypoints_{}{}.json'.
                format(split, year)))
        self.catIds = self.coco.getCatIds(catNms=['person'])
        self.imgIds = sorted(self.coco.getImgIds(catIds=self.catIds))

    def __len__(self):
        return len(self.imgIds)

    def generate_masks(self, img, annotations):
        mask_all = np.zeros(img.shape[:2], 'bool')
        mask_miss = np.zeros(img.shape[:2], 'bool')
        for ann in annotations:
            mask = self.coco.annToMask(ann).astype('bool')
            if ann['iscrowd'] == 1:
                intxn = mask_all & mask
                mask_miss = mask_miss | np.bitwise_xor(mask, intxn)
                mask_all = mask_all | mask
            elif (ann['num_keypoints'] < self.min_keypoints) or \
                 (ann['area'] <= self.min_area):
                mask_all = mask_all | mask
                mask_miss = mask_miss | mask
            else:
                mask_all = mask_all | mask
        return mask_all, mask_miss

    def get_image_annotation(self, index=None, img_id=None):
        annotations = None
        ignore_mask = None
        if index is not None:
            img_id = self.imgIds[index]
        if img_id is None:
            raise ValueError
        anno_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)

        # load image
        img_path = osp.join(
            self.img_root,
            self.coco.loadImgs([img_id])[0]['file_name'])
        img = cv2.imread(img_path)

        # annotation for that image
        if len(anno_ids) > 0:
            annotations_for_img = self.coco.loadAnns(anno_ids)

            _, ignore_mask = self.generate_masks(img, annotations_for_img)

            person_cnt = 0
            valid_annotations_for_img = []
            for annotation in annotations_for_img:
                # if too few keypoints or too small
                if annotation['num_keypoints'] >= self.min_keypoints and \
                   annotation['area'] > self.min_area:
                    person_cnt += 1
                    valid_annotations_for_img.append(annotation)

            # if person annotation
            if person_cnt > 0:
                annotations = valid_annotations_for_img

        if ignore_mask is None:
            ignore_mask = np.zeros(img.shape[:2], 'bool')

        return img, img_id, annotations, ignore_mask

    def parse_coco_annotation(self, annotations):
        """Convert coco annotation data to poses"""
        poses = np.zeros((0, len(JointType), 3), dtype=np.int32)

        for ann in annotations:
            ann_pose = np.array(ann['keypoints']).reshape(-1, 3)
            pose = np.zeros((1, len(JointType), 3), dtype=np.int32)

            # convert poses position
            for i, joint_index in enumerate(coco_utils.coco_joint_indices):
                pose[0][joint_index] = ann_pose[i]

            # compute neck position
            if pose[0][JointType.LeftShoulder][2] > 0 and \
               pose[0][JointType.RightShoulder][2] > 0:
                pose[0][JointType.Neck][0] = \
                    int((pose[0][JointType.LeftShoulder][0] +
                         pose[0][JointType.RightShoulder][0]) / 2)
                pose[0][JointType.Neck][1] = \
                    int((pose[0][JointType.LeftShoulder][1] +
                         pose[0][JointType.RightShoulder][1]) / 2)
                pose[0][JointType.Neck][2] = 2
            poses = np.vstack((poses, pose))
        return poses

    def get_example(self, i):
        img, img_id, annotations, ignore_mask = self.get_image_annotation(i)

        # if no annotations are available
        while annotations is None:
            img_id = self.imgIds[np.random.randint(len(self))]
            img, img_id, annotations, ignore_mask = self.get_image_annotation(
                img_id=img_id)

        poses = self.parse_coco_annotation(annotations)
        return img, poses, ignore_mask


if __name__ == '__main__':
    from chainer_openpose.visualizations import overlay_ignore_mask
    from chainer_openpose.visualizations import overlay_pafs
    from chainer_openpose.visualizations import overlay_heatmaps
    from chainer_openpose.transforms.heatmap import generate_heatmaps
    from chainer_openpose.transforms.paf import generate_pafs

    dataset = COCOPersonKeypointsDataset()

    index = 0
    while index < len(dataset):
        img, poses, ignore_mask = dataset[index]
        img = overlay_ignore_mask(img, ignore_mask)
        heatmaps = generate_heatmaps(img, poses, sigma=1)
        pafs = generate_pafs(img, poses, coco_utils.coco_joint_pairs, sigma=1)
        img = overlay_heatmaps(img, heatmaps)
        img = overlay_pafs(img, pafs)
        cv2.imshow(
            'coco',
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
