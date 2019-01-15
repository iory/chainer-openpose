import argparse

import chainer
from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions

from chainer_openpose.extensions import OpenPoseVisReport
from chainer_openpose.links import OpenPoseNet
from chainer_openpose.links import OpenPoseTrainChain
from chainer_openpose.datasets import COCOPersonKeypointsDataset
from chainer_openpose.transforms.heatmap import generate_heatmaps
from chainer_openpose.transforms.paf import generate_pafs
from chainer_openpose.transforms import resize
from chainer_openpose.transforms import random_resize
from chainer_openpose.transforms import random_rotate
from chainer_openpose.transforms import random_crop
from chainer_openpose.transforms import random_flip
from chainer_openpose.transforms import distort_color
from chainer_openpose.datasets.coco import coco_utils
from chainer_openpose.utils import prepare_output_dir
from pose_detector import PoseDetector


def copy_vgg_params(model):
    from chainer.links import caffe

    from chainer_openpose.datasets.coco.coco_utils import \
        get_vgg_pretrained_model

    print('Copying params of pretrained model...')
    layer_names = [
        "conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1",
        "conv3_2", "conv3_3", "conv3_4", "conv4_1", "conv4_2",
    ]

    pre_model = caffe.CaffeFunction(get_vgg_pretrained_model())
    for layer_name in layer_names:
        getattr(model, layer_name).W.data = pre_model[layer_name].W.data
        getattr(model, layer_name).b.data = pre_model[layer_name].b.data
    print('Done.')


class GradientScaling(object):

    name = 'GradientScaling'

    def __init__(self, layer_names, scale):
        self.layer_names = layer_names
        self.scale = scale

    def __call__(self, opt):
        for layer_name in self.layer_names:
            for param in opt.target[layer_name].params(False):
                grad = param.grad
                with chainer.cuda.get_device_from_array(grad):
                    grad *= self.scale


class Transform(object):

    def __init__(self, mode='train'):
        self.mode = mode
        self.input_size = (368, 368)

    def __call__(self, in_data):
        img, poses, ignore_mask = in_data
        if self.mode == 'train':
            img, ignore_mask, poses = random_resize(
                img, ignore_mask, poses)
            img, ignore_mask, poses = random_rotate(
                img, ignore_mask, poses)
            img, ignore_mask, poses = random_crop(
                img, ignore_mask, poses)
            img = distort_color(img)
            img, ignore_mask, poses = random_flip(
                img, ignore_mask, poses)
        img, ignore_mask, poses = resize(
            img, ignore_mask, poses, self.input_size)
        heatmaps = generate_heatmaps(img, poses)
        pafs = generate_pafs(img, poses, coco_utils.coco_joint_pairs)
        img = (img.astype('f') / 255.0) - 0.5
        img = img.transpose(2, 0, 1)  # hwc => chw
        return img, pafs, heatmaps, ignore_mask


def main():
    parser = argparse.ArgumentParser(description='Train pose estimation')
    parser.add_argument('--batchsize', '-B', type=int, default=10,
                        help='Training minibatch size')
    parser.add_argument('--valbatchsize', '-b', type=int, default=4,
                        help='Validation minibatch size')
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--vis-interval', type=int, default=20)
    parser.add_argument('--val-interval', type=int, default=1000)
    parser.add_argument('--val-samples', type=int, default=100,
                        help='Number of validation samples')
    parser.add_argument('--iteration', '-i', type=int, default=300000,
                        help='Number of iterations to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--domain-randomization', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()
    result_output_path = prepare_output_dir(args, args.out)
    print("output file: {}".format(result_output_path))

    model = OpenPoseNet(len(coco_utils.JointType) + 1,
                        len(coco_utils.coco_joint_pairs) * 2)
    copy_vgg_params(model)
    train_chain = OpenPoseTrainChain(model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        train_chain.to_gpu()

    optimizer = chainer.optimizers.Adam(alpha=1e-4,
                                        beta1=0.9,
                                        beta2=0.999,
                                        eps=1e-08)
    optimizer.setup(train_chain)

    layer_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
                   'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2']
    for layer_name in layer_names:
        train_chain.model[layer_name].disable_update()

    layer_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
                   'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2',
                   'conv4_3_CPM', 'conv4_4_CPM']
    optimizer.add_hook(GradientScaling(layer_names, 1.0 / 4.0))

    train_datasets = COCOPersonKeypointsDataset(split='train')
    train = TransformDataset(train_datasets, Transform(mode='train'))
    val_datasets = COCOPersonKeypointsDataset(split='val')
    val = TransformDataset(val_datasets, Transform(mode='val'))

    if args.loaderjob:
        train_iter = chainer.iterators.MultiprocessIterator(
            train, args.batchsize, n_processes=args.loaderjob)
    else:
        train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val_iter = chainer.iterators.SerialIterator(val, args.batchsize)

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater,
                               (args.iteration, 'iteration'),
                               result_output_path)

    val_interval = (args.val_interval, 'iteration')
    log_interval = (args.log_interval, 'iteration')
    vis_interval = (args.vis_interval, 'iteration')

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration',
        'main/loss', 'val/loss',
        'main/paf', 'val/paf',
        'main/heatmap', 'val/heatmap',
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=1))

    # log plotter
    trainer.extend(extensions.PlotReport([
        'main/loss', 'val/loss',
        'main/paf', 'val/paf',
        'main/heatmap', 'val/heatmap'], file_name='loss.png',
                                         trigger=log_interval),
                   trigger=log_interval)

    # Visualization.
    pose_detector = PoseDetector(model, device=args.gpu)
    trainer.extend(
        OpenPoseVisReport(
            val_iter,
            pose_detector,
            coco_utils.coco_joint_pairs,
            coco_utils.coco_skip_joint_pair_indices),
        trigger=vis_interval)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    trainer.run()


if __name__ == '__main__':
    main()
