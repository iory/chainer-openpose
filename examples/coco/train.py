import argparse

import chainer
from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions

from chainer_openpose.links import OpenPoseNet
from chainer_openpose.links import OpenPoseTrainChain
from chainer_openpose.datasets import COCOPersonKeypointsDataset
from chainer_openpose.transforms.heatmap import generate_heatmaps
from chainer_openpose.transforms.paf import generate_pafs
from chainer_openpose.transforms.resize import resize
from chainer_openpose.datasets.coco import coco_utils
from chainer_openpose.utils import prepare_output_dir


class Transform(object):

    def __init__(self, mode='train'):
        self.mode = mode
        self.input_size = (368, 368)

    def __call__(self, in_data):
        img, poses, ignore_mask = in_data
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
    args.out = prepare_output_dir(args, args.out)

    model = OpenPoseNet(len(coco_utils.JointType) + 1,
                        len(coco_utils.coco_joint_pairs) * 2)
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

    train_datasets = COCOPersonKeypointsDataset(split='train')
    train = TransformDataset(train_datasets, Transform(mode='train'))

    if args.loaderjob:
        train_iter = chainer.iterators.MultiprocessIterator(
            train, args.batchsize, n_processes=args.loaderjob)
    else:
        train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater,
                               (args.iteration, 'iteration'),
                               args.out)

    val_interval = (10 if args.test else 1000), 'iteration'
    log_interval = (1 if args.test else 20), 'iteration'

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

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    trainer.run()


if __name__ == '__main__':
    main()
