import copy
import os.path as osp
import shutil

import chainer
import cv2
import fcn
import six
import numpy as np

from chainer_openpose.visualizations import overlay_pose
from chainer_openpose.utils import makedirs


class OpenPoseVisReport(chainer.training.extensions.Evaluator):

    def __init__(self,
                 iterator,
                 target,
                 joint_index_pairs,
                 skip_connection_indices=[],
                 file_name='visualizations/iteration=%08d.jpg',
                 shape=(4, 4),
                 copy_latest=True):
        super(OpenPoseVisReport, self).__init__(iterator, target)
        self.file_name = file_name
        self._shape = shape
        self._copy_latest = copy_latest
        self.skip_connection_indices = skip_connection_indices
        self.joint_index_pairs = joint_index_pairs

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
        i = 0
        for batch in it:
            for sample in batch:
                img = sample[0]
                img = img.transpose(1, 2, 0)
                pose, score = target(img)
                img = np.asarray((img + 0.5) * 255.0, dtype=np.uint8)
                imgs.append(img)
                pred_scores.append(score)
                pred_poses.append(pose)
                i += 1
                if i >= (self._shape[0] * self._shape[1]):
                    break
            if i >= (self._shape[0] * self._shape[1]):
                break

        # visualize
        vizs = []
        for img, pose, score in six.moves.zip(imgs, pred_poses, pred_scores):
            pred_viz = overlay_pose(img, pose,
                                    self.joint_index_pairs,
                                    self.skip_connection_indices)
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
