import pkg_resources


__version__ = pkg_resources.get_distribution(
    'chainer_openpose').version


from chainer_openpose import links  # NOQA
