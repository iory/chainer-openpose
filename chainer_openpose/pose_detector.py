# -*- coding: utf-8 -*-
# this code is based on
# https://github.com/DeNA/Chainer_Realtime_Multi-Person_Pose_Estimation

from __future__ import division

import cv2
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter

import chainer
from chainer import cuda
import chainer.functions as F


class PoseDetector(object):

    def __init__(self,
                 model,
                 joint_type,
                 joint_pairs,
                 device=-1,
                 precise=False,
                 ksize=17,
                 gaussian_sigma=2.5,
                 heatmap_peak_thresh=0.1,
                 heatmap_size=320,
                 inference_scales=[0.38, 0.7, 1.0],
                 inference_img_size=368,
                 n_subset_limbs_thresh=7,
                 subset_score_thresh=0.4,
                 n_integ_points=10,
                 n_integ_points_thresh=8,
                 length_penalty_ratio=0.5,
                 length_penalty_value=1.0,
                 inner_product_thresh=0.05,
                 downscale=8):
        self.precise = precise
        self.model = model
        self.joint_type = joint_type
        self.joint_pairs = joint_pairs
        self.downscale = downscale

        self.device = device
        self.ksize = ksize
        self.gaussian_sigma = gaussian_sigma
        self.heatmap_peak_thresh = heatmap_peak_thresh
        self.heatmap_size = heatmap_size
        self.inference_scales = inference_scales
        self.inference_img_size = inference_img_size
        self.n_subset_limbs_thresh = n_subset_limbs_thresh
        self.subset_score_thresh = subset_score_thresh
        self.n_integ_points = n_integ_points
        self.n_integ_points_thresh = n_integ_points_thresh
        self.length_penalty_ratio = length_penalty_ratio
        self.length_penalty_value = length_penalty_value
        self.inner_product_thresh = inner_product_thresh
        if self.device >= 0:
            cuda.get_device_from_id(device).use()
            self.model.to_gpu()

            # create gaussian filter
            self.gaussian_kernel = self.create_gaussian_kernel(
                gaussian_sigma, ksize)[None, None]
            self.gaussian_kernel = cuda.to_gpu(self.gaussian_kernel)

    def create_gaussian_kernel(self, sigma=1, ksize=5):
        center = int(ksize / 2)
        grid_x = np.tile(np.arange(ksize), (ksize, 1))
        grid_y = grid_x.transpose().copy()
        grid_d2 = (grid_x - center) ** 2 + (grid_y - center) ** 2
        kernel = 1 / (sigma**2 * 2 * np.pi) * np.exp(-0.5 * grid_d2 / sigma**2)
        return kernel.astype('f')

    def pad_image(self, img, stride, pad_value):
        h, w, _ = img.shape

        pad = [0] * 2
        pad[0] = (stride - (h % stride)) % stride  # down
        pad[1] = (stride - (w % stride)) % stride  # right

        img_padded = np.zeros(
            (h + pad[0], w + pad[1], 3), 'uint8') + pad_value
        img_padded[:h, :w, :] = img.copy()
        return img_padded, pad

    def compute_optimal_size(self, orig_img, img_size, stride=8):
        orig_img_h, orig_img_w, _ = orig_img.shape
        aspect = orig_img_h / orig_img_w
        if orig_img_h < orig_img_w:
            img_h = img_size
            img_w = np.round(img_size / aspect).astype(int)
            surplus = img_w % stride
            if surplus != 0:
                img_w += stride - surplus
        else:
            img_w = img_size
            img_h = np.round(img_size * aspect).astype(int)
            surplus = img_h % stride
            if surplus != 0:
                img_h += stride - surplus
        return (img_w, img_h)

    def compute_peaks_from_heatmaps(self, heatmaps):
        """Computer peaks

        all_peaks: shape = [N, 5], column = (jointtype, x, y, score, index)
        """
        heatmaps = heatmaps[:-1]

        xp = cuda.get_array_module(heatmaps)

        if xp == np:
            all_peaks = []
            peak_counter = 0
            for i, heatmap in enumerate(heatmaps):
                heatmap = gaussian_filter(heatmap, sigma=self.gaussian_sigma)
                map_left = xp.zeros(heatmap.shape)
                map_right = xp.zeros(heatmap.shape)
                map_top = xp.zeros(heatmap.shape)
                map_bottom = xp.zeros(heatmap.shape)
                map_left[1:, :] = heatmap[:-1, :]
                map_right[:-1, :] = heatmap[1:, :]
                map_top[:, 1:] = heatmap[:, :-1]
                map_bottom[:, :-1] = heatmap[:, 1:]

                peaks_binary = xp.logical_and.reduce((
                    heatmap > self.heatmap_peak_thresh,
                    heatmap > map_left,
                    heatmap > map_right,
                    heatmap > map_top,
                    heatmap > map_bottom,
                ))

                peaks = zip(
                    xp.nonzero(peaks_binary)[1],
                    xp.nonzero(peaks_binary)[0])
                peaks_with_score = [
                    (i,)
                    + peak_pos
                    + (heatmap[peak_pos[1], peak_pos[0]],)
                    for peak_pos in peaks]
                peaks_id = range(
                    peak_counter,
                    peak_counter +
                    len(peaks_with_score))
                peaks_with_score_and_id = [peaks_with_score[i] + (peaks_id[i],)
                                           for i in range(len(peaks_id))]
                peak_counter += len(peaks_with_score_and_id)
                all_peaks.append(peaks_with_score_and_id)
            all_peaks = xp.array([peak for peaks_each_category in all_peaks
                                  for peak in peaks_each_category])
        else:
            heatmaps = F.convolution_2d(
                heatmaps[:, None], self.gaussian_kernel,
                stride=1, pad=int(self.ksize / 2)).data.squeeze()
            left_heatmaps = cuda.to_gpu(
                xp.zeros(heatmaps.shape), device=self.device)
            right_heatmaps = cuda.to_gpu(
                xp.zeros(heatmaps.shape), device=self.device)
            top_heatmaps = cuda.to_gpu(
                xp.zeros(
                    heatmaps.shape),
                device=self.device)
            bottom_heatmaps = cuda.to_gpu(
                xp.zeros(heatmaps.shape), device=self.device)
            left_heatmaps[:, 1:, :] = heatmaps[:, :-1, :]
            right_heatmaps[:, :-1, :] = heatmaps[:, 1:, :]
            top_heatmaps[:, :, 1:] = heatmaps[:, :, :-1]
            bottom_heatmaps[:, :, :-1] = heatmaps[:, :, 1:]

            peaks_binary = xp.logical_and(
                heatmaps > self.heatmap_peak_thresh,
                heatmaps >= right_heatmaps)
            peaks_binary = xp.logical_and(
                peaks_binary, heatmaps >= top_heatmaps)
            peaks_binary = xp.logical_and(
                peaks_binary, heatmaps >= left_heatmaps)
            peaks_binary = xp.logical_and(
                peaks_binary, heatmaps >= bottom_heatmaps)

            peak_c, peak_y, peak_x = xp.nonzero(peaks_binary)
            peak_score = heatmaps[peak_c, peak_y, peak_x]
            all_peaks = xp.vstack(
                (peak_c, peak_x, peak_y, peak_score)).transpose()
            all_peaks = xp.hstack(
                (all_peaks, xp.arange(len(all_peaks)).reshape(-1, 1)))
            all_peaks = all_peaks.get()
        return all_peaks

    def compute_candidate_connections(self, paf, cand_a, cand_b, img_len):
        candidate_connections = []
        for joint_a in cand_a:
            for joint_b in cand_b:
                vector = joint_b[:2] - joint_a[:2]
                norm = np.linalg.norm(vector)
                if norm == 0:
                    continue

                ys = np.linspace(
                    joint_a[1],
                    joint_b[1],
                    num=self.n_integ_points)
                xs = np.linspace(
                    joint_a[0],
                    joint_b[0],
                    num=self.n_integ_points)
                integ_points = np.stack([ys, xs]).T.round().astype('i')
                paf_in_edge = np.hstack([paf[0][np.hsplit(integ_points, 2)],
                                         paf[1][np.hsplit(integ_points, 2)]])
                unit_vector = vector / norm
                inner_products = np.dot(paf_in_edge, unit_vector)

                integ_value = inner_products.sum() / len(inner_products)
                integ_value_with_dist_prior = integ_value + min(
                    self.length_penalty_ratio * img_len / norm
                    - self.length_penalty_value, 0)

                n_valid_points = sum(
                    inner_products > self.inner_product_thresh)
                if n_valid_points > self.n_integ_points_thresh \
                        and integ_value_with_dist_prior > 0:
                    candidate_connections.append(
                        [int(joint_a[3]), int(joint_b[3]),
                         integ_value_with_dist_prior])
        candidate_connections = sorted(
            candidate_connections,
            key=lambda x: x[2],
            reverse=True)
        return candidate_connections

    def compute_connections(self, pafs, all_peaks, img_len):
        all_connections = []
        for i in range(len(self.joint_pairs)):
            paf_index = [i * 2, i * 2 + 1]
            paf = pafs[paf_index]
            limb_point = self.joint_pairs[i]
            cand_a = all_peaks[all_peaks[:, 0] == limb_point[0]][:, 1:]
            cand_b = all_peaks[all_peaks[:, 0] == limb_point[1]][:, 1:]

            if len(cand_a) > 0 and len(cand_b) > 0:
                candidate_connections = self.compute_candidate_connections(
                    paf, cand_a, cand_b, img_len)
                connections = np.zeros((0, 3))
                for index_a, index_b, score in candidate_connections:
                    if index_a not in connections[:, 0] \
                            and index_b not in connections[:, 1]:
                        connections = np.vstack(
                            [connections, [index_a, index_b, score]])
                        if len(connections) >= min(len(cand_a), len(cand_b)):
                            break
                all_connections.append(connections)
            else:
                all_connections.append(np.zeros((0, 3)))
        return all_connections

    def grouping_key_points(self, all_connections, candidate_peaks):
        n = len(self.joint_type) + 2
        subsets = -1 * np.ones((0, n))

        for l, connections in enumerate(all_connections):
            joint_a, joint_b = self.joint_pairs[l]

            for ind_a, ind_b, score in connections[:, :3]:
                ind_a, ind_b = int(ind_a), int(ind_b)

                joint_found_cnt = 0
                joint_found_subset_index = [-1, -1]
                for subset_ind, subset in enumerate(subsets):
                    if subset[joint_a] == ind_a or subset[joint_b] == ind_b:
                        joint_found_subset_index[joint_found_cnt] = subset_ind
                        joint_found_cnt += 1

                if joint_found_cnt == 1:
                    found_subset = subsets[joint_found_subset_index[0]]
                    if found_subset[joint_b] != ind_b:
                        found_subset[joint_b] = ind_b
                        found_subset[-1] += 1
                        found_subset[-2] += candidate_peaks[ind_b, 3] + score

                elif joint_found_cnt == 2:
                    found_subset_1 = subsets[joint_found_subset_index[0]]
                    found_subset_2 = subsets[joint_found_subset_index[1]]

                    membership = ((found_subset_1 >= 0).astype(int)
                                  + (found_subset_2 >= 0).astype(int))[:-2]
                    if not np.any(membership == 2):
                        found_subset_1[:-2] += found_subset_2[:-2] + 1
                        found_subset_1[-2:] += found_subset_2[-2:]
                        found_subset_1[-2:] += score
                        subsets = np.delete(
                            subsets, joint_found_subset_index[1], axis=0)
                    else:
                        if found_subset_1[joint_a] == -1:
                            found_subset_1[joint_a] = ind_a
                            found_subset_1[-1] += 1
                            found_subset_1[-2] += candidate_peaks[ind_a, 3] + score
                        elif found_subset_1[joint_b] == -1:
                            found_subset_1[joint_b] = ind_b
                            found_subset_1[-1] += 1
                            found_subset_1[-2] += candidate_peaks[ind_b, 3] + score
                        if found_subset_2[joint_a] == -1:
                            found_subset_2[joint_a] = ind_a
                            found_subset_2[-1] += 1
                            found_subset_2[-2] += candidate_peaks[ind_a, 3] + score
                        elif found_subset_2[joint_b] == -1:
                            found_subset_2[joint_b] = ind_b
                            found_subset_2[-1] += 1
                            found_subset_2[-2] += candidate_peaks[ind_b, 3] + score

                elif joint_found_cnt == 0:
                    row = -1 * np.ones(n)
                    row[joint_a] = ind_a
                    row[joint_b] = ind_b
                    row[-1] = 2
                    row[-2] = sum(candidate_peaks[[ind_a, ind_b], 3]) + score
                    subsets = np.vstack([subsets, row])
                elif joint_found_cnt >= 3:
                    pass

        # delete low score subsets
        keep = np.logical_and(
            subsets[:, - 1] >= self.n_subset_limbs_thresh,
            subsets[:, - 2] /
            subsets[:, - 1] >= self.subset_score_thresh)
        subsets = subsets[keep]
        return subsets

    def subsets_to_pose_array(self, subsets, all_peaks):
        n = len(self.joint_type)
        person_pose_array = []
        for subset in subsets:
            joints = []
            for joint_index in subset[:n].astype('i'):
                if joint_index >= 0:
                    joint = all_peaks[joint_index][1:3].tolist()
                    joint.append(2)
                    joint.append(float(all_peaks[joint_index][3]))
                    joints.append(joint)
                else:
                    joints.append([0, 0, 0, 0])
            person_pose_array.append(np.array(joints))
        person_pose_array = np.array(person_pose_array)
        return person_pose_array

    def detect_precise(self, orig_img):
        orig_img_h, orig_img_w, _ = orig_img.shape

        pafs_sum = 0
        heatmaps_sum = 0

        interpolation = cv2.INTER_CUBIC

        for scale in self.inference_scales:
            multiplier = scale * self.inference_img_size / \
                min(orig_img.shape[:2])
            img = cv2.resize(
                orig_img,
                (math.ceil(
                    orig_img_w *
                    multiplier),
                    math.ceil(
                    orig_img_h *
                    multiplier)),
                interpolation=interpolation)
            padded_img, pad = self.pad_image(
                img, self.downscale, (104, 117, 123))

            x_data = self.preprocess(padded_img)
            if self.device >= 0:
                x_data = cuda.to_gpu(x_data, device=self.device)
            x_data = x_data.astype('f')

            h1s, h2s = self.model(x_data)

            tmp_paf = h1s[-1][0].data.transpose(1, 2, 0)
            tmp_heatmap = h2s[-1][0].data.transpose(1, 2, 0)

            if self.device >= 0:
                tmp_paf = cuda.to_cpu(tmp_paf)
                tmp_heatmap = cuda.to_cpu(tmp_heatmap)

            p_h, p_w = padded_img.shape[:2]
            tmp_paf = cv2.resize(
                tmp_paf, (p_w, p_h), interpolation=interpolation)
            tmp_paf = tmp_paf[:p_h - pad[0], :p_w - pad[1], :]
            pafs_sum += cv2.resize(
                tmp_paf,
                (orig_img_w, orig_img_h),
                interpolation=interpolation)

            tmp_heatmap = cv2.resize(
                tmp_heatmap,
                (0,
                 0),
                fx=self.downscale,
                fy=self.downscale,
                interpolation=interpolation)
            tmp_heatmap = tmp_heatmap[:padded_img.shape[0] -
                                      pad[0], :padded_img.shape[1] - pad[1], :]
            heatmaps_sum += cv2.resize(tmp_heatmap,
                                       (orig_img_w,
                                        orig_img_h),
                                       interpolation=interpolation)

        self.pafs = (pafs_sum / len(self.inference_scales)).transpose(2, 0, 1)
        self.heatmaps = (heatmaps_sum /
                         len(self.inference_scales)).transpose(2, 0, 1)

        if self.device >= 0:
            self.pafs = cuda.to_cpu(self.pafs)

        self.all_peaks = self.compute_peaks_from_heatmaps(self.heatmaps)
        if len(self.all_peaks) == 0:
            return np.empty((0, len(self.joint_type), 3)), np.empty(0), None
        all_connections = self.compute_connections(
            self.pafs, self.all_peaks, orig_img_w)
        subsets = self.grouping_key_points(all_connections, self.all_peaks)
        poses = self.subsets_to_pose_array(subsets, self.all_peaks)
        scores = subsets[:, -2]
        return poses, scores, self.all_peaks

    def preprocess(self, x):
        x = x.astype('f')
        x /= 255
        x -= 0.5
        x = x.transpose(2, 0, 1)
        return x[None, ]

    def predict(self, imgs):
        """Detect poses from images.

        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in HWC and BGR format
                and the range of their value is :math:`[0, 255]`.

        """
        pose = []
        score = []
        with chainer.using_config('train', False), \
                chainer.function.no_backprop_mode():
            for img in imgs:
                poses, scores = self(img)
                pose.append(poses)
                score.append(scores)
        return pose, score

    def __call__(self, orig_img):
        orig_img = orig_img.copy()
        if self.precise:
            return self.detect_precise(orig_img)
        orig_img_h, orig_img_w, _ = orig_img.shape

        input_w, input_h = self.compute_optimal_size(
            orig_img, self.inference_img_size)
        map_w, map_h = self.compute_optimal_size(orig_img, self.heatmap_size)

        resized_image = cv2.resize(orig_img, (input_w, input_h))
        x_data = self.preprocess(resized_image)

        if self.device >= 0:
            x_data = cuda.to_gpu(x_data, device=self.device)

        h1s, h2s = self.model(x_data)

        pafs = F.resize_images(h1s[-1], (map_h, map_w)).data[0]
        heatmaps = F.resize_images(h2s[-1], (map_h, map_w)).data[0]

        if self.device >= 0:
            pafs = pafs.get()
            heatmaps = heatmaps.get()
            cuda.get_device_from_id(self.device).synchronize()

        all_peaks = self.compute_peaks_from_heatmaps(heatmaps)
        if len(all_peaks) == 0:
            return np.empty((0, len(self.joint_type), 3)), np.empty(0), None
        all_connections = self.compute_connections(pafs, all_peaks, map_w)
        subsets = self.grouping_key_points(all_connections, all_peaks)
        all_peaks[:, 1] *= orig_img_w / map_w
        all_peaks[:, 2] *= orig_img_h / map_h
        poses = self.subsets_to_pose_array(subsets, all_peaks)
        scores = subsets[:, -2]
        return poses, scores, all_peaks
