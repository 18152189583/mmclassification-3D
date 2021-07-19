import os

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS
from .pipelines import Compose


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_folders(root):
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        folder_to_idx (dict): the map from folder name to class idx
    """
    folders = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


# def get_samples(root, folder_to_idx, extensions):
#     """Make dataset by walking all images under a root.
#
#     Args:
#         root (string): root directory of folders
#         folder_to_idx (dict): the map from class name to class idx
#         extensions (tuple): allowed extensions
#
#     Returns:
#         samples (list): a list of tuple where each element is (image, label)
#     """
#     samples = []
#     root = os.path.expanduser(root)
#     for folder_name in sorted(os.listdir(root)):
#         _dir = os.path.join(root, folder_name)
#         if not os.path.isdir(_dir):
#             continue
#
#         for _, _, fns in sorted(os.walk(_dir)):
#             for fn in sorted(fns):
#                 if has_file_allowed_extension(fn, extensions):
#                     path = os.path.join(folder_name, fn)
#                     item = (path, folder_to_idx[folder_name])
#                     samples.append(item)
#     return samples


@DATASETS.register_module()
class Hie_Dataset(BaseDataset):

    IMG_EXTENSIONS = ('.nii.gz')
    CLASSES = [
        '0', '1'
    ]

    def __init__(self,
                 data_prefix,
                 pipeline,
                 classes=None,
                 ann_file=None,
                 modes=[],
                 test_mode=False):
        super(BaseDataset, self).__init__()

        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.CLASSES = self.get_classes(classes)
        self.modes = modes
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        if self.ann_file is None:
            print("[ERROR] need label info: {}".format(self.__class__.__name__))
        elif isinstance(self.ann_file, str):
            if self.ann_file.endswith('.txt'):
                with open(self.ann_file) as f:
                    samples = [x.strip().split(' ') for x in f.readlines()]
            else:
                raise NotImplementedError
        else:
            raise TypeError('ann_file must be a str or None')
        self.samples = samples

        data_infos = []
        for filename, gt_label in self.samples:
            info = {'img_prefix': self.data_prefix}
            filenames = []
            for mode in self.modes:
                filenames.append(os.path.join(self.data_prefix, filename, mode + '.nii.gz'))
            info['img_info'] = {'filename': filenames}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos
