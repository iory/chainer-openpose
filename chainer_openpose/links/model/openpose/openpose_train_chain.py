import numpy as np
import chainer
import chainer.functions as F


class OpenPoseTrainChain(chainer.Chain):

    def __init__(self, model):
        super(OpenPoseTrainChain, self).__init__()
        with self.init_scope():
            self.model = model

    def compute_loss(self,
                     images,
                     pafs_ys,
                     heatmaps_ys,
                     ground_truth_pafs,
                     ground_truth_heatmaps,
                     ignore_mask):
        """

        ground_truth_pafs : list of grount truth paf

        """
        heatmap_losses = []
        paf_losses = []
        loss = 0.0

        paf_masks = ignore_mask[:, None].repeat(
            ground_truth_pafs.shape[1], axis=1)
        heatmap_masks = ignore_mask[:, None].repeat(
            ground_truth_heatmaps.shape[1], axis=1)

        # compute loss on each stage
        for pafs_y, heatmaps_y in zip(pafs_ys, heatmaps_ys):
            stage_ground_truth_pafs = ground_truth_pafs.copy()
            stage_ground_truth_heatmaps = ground_truth_heatmaps.copy()
            stage_paf_masks = paf_masks.copy()
            stage_heatmap_masks = heatmap_masks.copy()

            if pafs_y.shape != stage_ground_truth_pafs.shape:
                stage_ground_truth_pafs = F.resize_images(
                    stage_ground_truth_pafs, pafs_y.shape[2:]).data
                stage_ground_truth_heatmaps = F.resize_images(
                    stage_ground_truth_heatmaps, pafs_y.shape[2:]).data
                stage_paf_masks = F.resize_images(
                    stage_paf_masks.astype('f'), pafs_y.shape[2:]).data > 0
                stage_heatmap_masks = F.resize_images(
                    stage_heatmap_masks.astype('f'), pafs_y.shape[2:]).data > 0

            stage_ground_truth_pafs[stage_paf_masks == True] = \
                pafs_y.data[stage_paf_masks == True]
            stage_ground_truth_heatmaps[stage_heatmap_masks == True] = \
                heatmaps_y.data[stage_heatmap_masks == True]

            pafs_loss = F.mean_squared_error(
                pafs_y, stage_ground_truth_pafs)
            heatmaps_loss = F.mean_squared_error(
                heatmaps_y, stage_ground_truth_heatmaps)

            loss += pafs_loss + heatmaps_loss

            paf_losses.append(
                float(chainer.cuda.to_cpu(pafs_loss.data)))
            heatmap_losses.append(
                float(chainer.cuda.to_cpu(heatmaps_loss.data)))

        return loss, paf_losses, heatmap_losses

    def forward(self,
                images,
                ground_truth_pafs,
                ground_truth_heatmaps,
                ignore_masks):
        """

        images : 4-D Tensors indexed by time.
              (batch_size, seq_channels, height, width)

        """
        pafs, heatmaps = self.model(images)
        loss, paf_losses, heatmap_losses = \
            self.compute_loss(images,
                              pafs,
                              heatmaps,
                              ground_truth_pafs,
                              ground_truth_heatmaps,
                              ignore_masks)

        chainer.reporter.report(
            {'loss': loss,
             'paf': sum(paf_losses),
             'heatmap': sum(heatmap_losses)}, self)

        return loss


if __name__ == '__main__':
    import cupy
    import numpy as np

    import chainer_openpose

    model = chainer_openpose.links.OpenPoseNet()
    train_chain = OpenPoseTrainChain(model)
