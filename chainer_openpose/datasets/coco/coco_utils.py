import os
from enum import IntEnum

import chainer
from chainer.dataset import download
from chainer.links import caffe
from chainercv import utils
from eos import makedirs

from chainer_openpose.links import OpenPoseNet


root = 'iory/openpose/coco'

img_urls = {
    '2017': {
        'train': 'http://images.cocodataset.org/zips/train2017.zip',
        'val': 'http://images.cocodataset.org/zips/val2017.zip'
    }
}

person_keypoints_anno_urls = {
    '2017': {
        'train': 'http://images.cocodataset.org/annotations/'
        'annotations_trainval2017.zip',
        'val': 'http://images.cocodataset.org/annotations/'
        'annotations_trainval2017.zip'
    }
}


def get_coco(split, year, mode='person_keypoints'):
    if year not in ['2017']:
        raise ValueError
    if split not in ['train', 'val']:
        raise ValueError
    data_dir = download.get_dataset_directory(root)
    annos_root = os.path.join(data_dir, 'annotations')
    img_root = os.path.join(data_dir, 'images')
    created_img_root = os.path.join(
        img_root, '{}{}'.format(split, year))
    img_url = img_urls[year][split]

    if mode == 'person_keypoints':
        anno_url = person_keypoints_anno_urls[year][split]
        anno_path = os.path.join(
            annos_root, 'person_keypoints_{}{}.json'.format(split, year))
    else:
        raise ValueError('invalid mode {}'.format(mode))

    if not os.path.exists(created_img_root):
        download_file_path = utils.cached_download(img_url)
        ext = os.path.splitext(img_url)[1]
        utils.extractall(download_file_path, img_root, ext)
    if not os.path.exists(anno_path):
        download_file_path = utils.cached_download(anno_url)
        ext = os.path.splitext(anno_url)[1]
        if split in ['train', 'val']:
            utils.extractall(download_file_path, data_dir, ext)
        elif split in ['valminusminival', 'minival']:
            utils.extractall(download_file_path, annos_root, ext)
    return data_dir


def get_coco_pretrained_model():
    data_dir = download.get_dataset_directory(root)
    pretraind_model_dir = os.path.join(data_dir, 'model')
    makedirs(pretraind_model_dir)
    pretrained_model_filename = os.path.join(
        pretraind_model_dir, 'coco_posenet.npz')
    pretrained_model_url = 'http://posefs1.perception.cs.cmu.edu/OpenPose/'\
        'models/pose/coco/pose_iter_440000.caffemodel'
    download_file_path = utils.cached_download(pretrained_model_url)
    if os.path.exists(pretrained_model_filename):
        return pretrained_model_filename
    model = OpenPoseNet(len(JointType) + 1, len(coco_joint_pairs) * 2)
    caffe_model = caffe.CaffeFunction(download_file_path)

    layer_names = [
        "conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2",
        "conv3_3", "conv3_4", "conv4_1", "conv4_2", "conv4_3_CPM",
        "conv4_4_CPM", "conv5_1_CPM_L1", "conv5_2_CPM_L1", "conv5_3_CPM_L1",
        "conv5_4_CPM_L1", "conv5_1_CPM_L2", "conv5_2_CPM_L2", "conv5_3_CPM_L2",
        "conv5_4_CPM_L2", "conv5_5_CPM_L2", "Mconv1_stage2_L1",
        "Mconv2_stage2_L1", "Mconv3_stage2_L1", "Mconv4_stage2_L1",
        "Mconv5_stage2_L1", "Mconv6_stage2_L1", "Mconv7_stage2_L1",
        "Mconv1_stage2_L2", "Mconv2_stage2_L2", "Mconv3_stage2_L2",
        "Mconv4_stage2_L2", "Mconv5_stage2_L2", "Mconv6_stage2_L2",
        "Mconv7_stage2_L2", "Mconv1_stage3_L1", "Mconv2_stage3_L1",
        "Mconv3_stage3_L1", "Mconv4_stage3_L1", "Mconv5_stage3_L1",
        "Mconv6_stage3_L1", "Mconv7_stage3_L1", "Mconv1_stage3_L2",
        "Mconv2_stage3_L2", "Mconv3_stage3_L2", "Mconv4_stage3_L2",
        "Mconv5_stage3_L2", "Mconv6_stage3_L2", "Mconv7_stage3_L2",
        "Mconv1_stage4_L1", "Mconv2_stage4_L1", "Mconv3_stage4_L1",
        "Mconv4_stage4_L1", "Mconv5_stage4_L1", "Mconv6_stage4_L1",
        "Mconv7_stage4_L1", "Mconv1_stage4_L2", "Mconv2_stage4_L2",
        "Mconv3_stage4_L2", "Mconv4_stage4_L2", "Mconv5_stage4_L2",
        "Mconv6_stage4_L2", "Mconv7_stage4_L2", "Mconv1_stage5_L1",
        "Mconv2_stage5_L1", "Mconv3_stage5_L1", "Mconv4_stage5_L1",
        "Mconv5_stage5_L1", "Mconv6_stage5_L1", "Mconv7_stage5_L1",
        "Mconv1_stage5_L2", "Mconv2_stage5_L2", "Mconv3_stage5_L2",
        "Mconv4_stage5_L2", "Mconv5_stage5_L2", "Mconv6_stage5_L2",
        "Mconv7_stage5_L2", "Mconv1_stage6_L1", "Mconv2_stage6_L1",
        "Mconv3_stage6_L1", "Mconv4_stage6_L1", "Mconv5_stage6_L1",
        "Mconv6_stage6_L1", "Mconv7_stage6_L1", "Mconv1_stage6_L2",
        "Mconv2_stage6_L2", "Mconv3_stage6_L2", "Mconv4_stage6_L2",
        "Mconv5_stage6_L2", "Mconv6_stage6_L2", "Mconv7_stage6_L2"]

    # copy layer params
    for layer_name in layer_names:
        model[layer_name].copyparams(caffe_model[layer_name])
    chainer.serializers.save_npz(pretrained_model_filename, model)
    return pretrained_model_filename


def get_vgg_pretrained_model():
    vgg_url = 'http://www.robots.ox.ac.uk/%7Evgg/software/'\
        'very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel'
    download_file_path = utils.cached_download(vgg_url)
    return download_file_path


class JointType(IntEnum):
    Nose = 0
    Neck = 1
    RightShoulder = 2
    RightElbow = 3
    RightHand = 4
    LeftShoulder = 5
    LeftElbow = 6
    LeftHand = 7
    RightWaist = 8
    RightKnee = 9
    RightFoot = 10
    LeftWaist = 11
    LeftKnee = 12
    LeftFoot = 13
    RightEye = 14
    LeftEye = 15
    RightEar = 16
    LeftEar = 17


coco_joint_indices = (JointType.Nose,
                      JointType.LeftEye,
                      JointType.RightEye,
                      JointType.LeftEar,
                      JointType.RightEar,
                      JointType.LeftShoulder,
                      JointType.RightShoulder,
                      JointType.LeftElbow,
                      JointType.RightElbow,
                      JointType.LeftHand,
                      JointType.RightHand,
                      JointType.LeftWaist,
                      JointType.RightWaist,
                      JointType.LeftKnee,
                      JointType.RightKnee,
                      JointType.LeftFoot,
                      JointType.RightFoot)


coco_joint_pairs = ((JointType.Neck, JointType.RightWaist),
                    (JointType.RightWaist, JointType.RightKnee),
                    (JointType.RightKnee, JointType.RightFoot),
                    (JointType.Neck, JointType.LeftWaist),
                    (JointType.LeftWaist, JointType.LeftKnee),
                    (JointType.LeftKnee, JointType.LeftFoot),
                    (JointType.Neck, JointType.RightShoulder),
                    (JointType.RightShoulder, JointType.RightElbow),
                    (JointType.RightElbow, JointType.RightHand),
                    (JointType.RightShoulder, JointType.RightEar),
                    (JointType.Neck, JointType.LeftShoulder),
                    (JointType.LeftShoulder, JointType.LeftElbow),
                    (JointType.LeftElbow, JointType.LeftHand),
                    (JointType.LeftShoulder, JointType.LeftEar),
                    (JointType.Neck, JointType.Nose),
                    (JointType.Nose, JointType.RightEye),
                    (JointType.Nose, JointType.LeftEye),
                    (JointType.RightEye, JointType.RightEar),
                    (JointType.LeftEye, JointType.LeftEar))


# don't show ear-shoulder connection
coco_skip_joint_pair_indices = (9, 13)
