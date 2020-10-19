import copy
import os.path as osp
import shutil

import chainer
import cv2
import fcn
import six
import numpy as np
import matplotlib
from eos import makedirs

from chainer_openpose.visualizations import overlay_pose


class OpenPoseVisReport(chainer.training.extensions.Evaluator):

    def __init__(self,
                 iterator,
                 target,
                 joint_index_pairs,
                 skip_connection_indices=[],
                 peak_threshold=0.5,
                 file_name='visualizations/iteration=%08d.jpg',
                 shape=(4, 4),
                 copy_latest=True):
        super(OpenPoseVisReport, self).__init__(iterator, target)
        self.file_name = file_name
        self._shape = shape
        self._copy_latest = copy_latest
        self.skip_connection_indices = skip_connection_indices
        self.joint_index_pairs = joint_index_pairs
        self.peak_threshold = peak_threshold

    def __call__(self, trainer):
        iterator = self._iterators['main']
        target = self._targets

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        imgs = []
        pred_poses = []
        pred_scores = []
        pred_all_peaks = []
        i = 0
        for batch in it:
            for sample in batch:
                img = sample[0]
                img = np.asarray(np.clip((img + 0.5) * 255.0, 0, 255),
                                 dtype=np.uint8)
                img = img.transpose(1, 2, 0)
                pose, score, all_peaks = target(img)
                imgs.append(img)
                pred_scores.append(score)
                pred_poses.append(pose)
                pred_all_peaks.append(all_peaks)
                i += 1
                if i >= (self._shape[0] * self._shape[1]):
                    break
            if i >= (self._shape[0] * self._shape[1]):
                break

        # visualize
        cmap = matplotlib.cm.get_cmap('hsv')
        vizs = []
        for img, pose, score, all_peaks in six.moves.zip(
                imgs, pred_poses, pred_scores, pred_all_peaks):
            # keypoints
            if all_peaks is not None:
                n = len(target.joint_type)
                for j in range(len(all_peaks)):
                    score = all_peaks[j][3]
                    if score < self.peak_threshold:
                        continue
                    i = all_peaks[j][0]
                    rgba = np.array(cmap(1. * i / n))
                    color = rgba[:3] * 255
                    cv2.circle(img,
                               (int(all_peaks[j][1]),
                                int(all_peaks[j][2])),
                               4, color, thickness=-1)

            if len(pose) > 0:
                pred_viz = overlay_pose(img, pose[:, :, :3],
                                        self.joint_index_pairs,
                                        self.skip_connection_indices)
            else:
                pred_viz = img.copy()
            vizs.append(pred_viz[:, :, ::-1])
            if len(vizs) >= (self._shape[0] * self._shape[1]):
                break

        viz = fcn.utils.get_tile_image(vizs, tile_shape=self._shape)
        file_name = osp.join(
            trainer.out, self.file_name % trainer.updater.iteration)

        makedirs(osp.dirname(file_name), exist_ok=True)
        cv2.imwrite(file_name, viz[:, :, ::-1])

        if self._copy_latest:
            shutil.copy(file_name,
                        osp.join(osp.dirname(file_name), 'latest.jpg'))
