import pkg_resources


__version__ = pkg_resources.get_distribution(
    'chainer-openpose').version


from chainer_openpose import datasets  # NOQA
from chainer_openpose import extensions  # NOQA
from chainer_openpose import links  # NOQA
from chainer_openpose import transforms  # NOQA
from chainer_openpose import visualizations  # NOQA

from chainer_openpose.pose_detector import PoseDetector  # NOQA
