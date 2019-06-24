import hashlib
import os
import shutil
import tempfile

import filelock
from chainer.dataset import download
import gdown


def cached_gdown_download(url, cached_path=None):
    cache_root = os.path.join(download.get_dataset_root(), '_dl_cache')
    try:
        os.makedirs(cache_root)
    except OSError:
        if not os.path.exists(cache_root):
            raise

    urlhash = hashlib.md5(url.encode('utf-8')).hexdigest()
    if cached_path is None:
        cached_path = os.path.join(cache_root, urlhash)
    lock_path = cached_path + ".lock"

    with filelock.FileLock(lock_path):
        if os.path.exists(cached_path):
            return cached_path

    temp_root = tempfile.mkdtemp(dir=cache_root)
    try:
        temp_path = os.path.join(temp_root, 'download.cache')
        gdown.download(url, temp_path, quiet=False)
        with filelock.FileLock(lock_path):
            shutil.move(temp_path, cached_path)
    finally:
        shutil.rmtree(temp_root)

    return cached_path
