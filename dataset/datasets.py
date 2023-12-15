import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import ConcatDataset, Subset
from PIL import Image, ImageFile
from torchvision import transforms
from learning.augmentations import get_image_aug

ImageFile.LOAD_TRUNCATED_IMAGES = True


DATASETS = [
    'MIMIC',
    'CheXpert',
    'NIH',
    'PadChest',
    'VinDr',
    'SIIM',
    'ISIC',
    'ODIR'
]

CXR_DATASETS = [
    'MIMIC',
    'CheXpert',
    'NIH',
    'PadChest',
    'VinDr',
    'SIIM'
]

ATTRS = ['sex', 'ethnicity', 'age', 'sex_ethnicity']
TASKS = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
         'Cataract', 'Retinopathy']


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError(f"Dataset not found: {dataset_name}")
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class SubpopDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    INPUT_SHAPE = None       # Subclasses should override
    AVAILABLE_ATTRS = None   # Subclasses should override
    SPLITS = {               # Default, subclasses may override
        'tr': 0,
        'va': 1,
        'te': 2
    }
    EVAL_SPLITS = ['te']     # Default, subclasses may override

    def __init__(self, root, split, metadata, transform, group_def='group', subsample_type=None, duplicates=None, subset_query=None):
        if metadata is not None:
            df = pd.read_csv(metadata)
            df = df[df["split"] == (self.SPLITS[split])]

            if subset_query is not None:
                df = df.query(subset_query)
                # since we drop group 0 for ISIC and ODIR, this hack prevents crashing later on
                if df['age'].min() == 1:
                    df['age'] = df['age'] - 1

            df['y'] = df[self.task_name]
            df['a'] = df[self.attr_name]
            self.idx = list(range(len(df)))
            self.x = df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
            self.y = df["y"].astype(int).tolist()
            self.a = df["a"].astype(int).tolist() if group_def in ['group', 'attr'] else [0] * len(df["a"].tolist())
            self.transform_ = transform
            self._count_groups()

            if subsample_type is not None:
                self.subsample(subsample_type)

            if duplicates is not None:
                self.duplicate(duplicates)

    def _count_groups(self):
        self.weights_g, self.weights_y, self.weights_a = [], [], []
        self.num_attributes = len(set(self.a))
        if self.num_attributes != max(set(self.a)) + 1:
            self.num_attributes = max(set(self.a)) + 1
        self.num_labels = len(set(self.y))
        self.group_sizes = [0] * self.num_attributes * self.num_labels
        self.class_sizes = [0] * self.num_labels
        self.attr_sizes = [0] * self.num_attributes

        for i in self.idx:
            self.group_sizes[self.num_attributes * self.y[i] + self.a[i]] += 1
            self.class_sizes[self.y[i]] += 1
            self.attr_sizes[self.a[i]] += 1

        for i in self.idx:
            self.weights_g.append(len(self) / self.group_sizes[self.num_attributes * self.y[i] + self.a[i]])
            self.weights_y.append(len(self) / self.class_sizes[self.y[i]])
            self.weights_a.append(len(self) / self.attr_sizes[self.a[i]])

    def subsample(self, subsample_type):
        assert subsample_type in {"group", "class"}
        perm = torch.randperm(len(self)).tolist()
        min_size = min([x for x in self.group_sizes if x > 0]) if subsample_type == "group" \
            else min(list(self.class_sizes))

        counts_g = [0] * self.num_attributes * self.num_labels
        counts_y = [0] * self.num_labels
        new_idx = []
        for p in perm:
            y, a = self.y[self.idx[p]], self.a[self.idx[p]]
            if (subsample_type == "group" and counts_g[self.num_attributes * int(y) + int(a)] < min_size) or (
                    subsample_type == "class" and counts_y[int(y)] < min_size):
                counts_g[self.num_attributes * int(y) + int(a)] += 1
                counts_y[int(y)] += 1
                new_idx.append(self.idx[p])

        self.idx = new_idx
        self._count_groups()

    def duplicate(self, duplicates):
        new_idx = []
        for i, duplicate in zip(self.idx, duplicates):
            new_idx += [i] * duplicate
        self.idx = new_idx
        self._count_groups()

    def __getitem__(self, index):
        i = self.idx[index]
        x = self.transform(self.x[i])
        y = torch.tensor(self.y[i], dtype=torch.long)
        a = torch.tensor(self.a[i], dtype=torch.long)
        return i, x, y, a

    def __len__(self):
        return len(self.idx)


class BaseImageDataset(SubpopDataset):

    def __init__(self, metadata, split, hparams, group_def='group', subsample_type=None, duplicates=None, override_attr=None, subset_query=None):
        transform = get_image_aug(hparams['data_augmentation'], self.INPUT_SHAPE[1:], 256) \
            if split == 'tr' and hparams['data_augmentation'] != 'none' else \
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        self.data_type = "images"

        self.task_name = hparams['task']
        self.attr_name = override_attr if override_attr is not None else hparams['attr']
        super().__init__('/', split, metadata, transform, group_def, subsample_type, duplicates, subset_query)

    def transform(self, x):
        if self.__class__.__name__ in ['MIMIC'] and 'MIMIC-CXR-JPG' in x:
            reduced_img_path = list(Path(x).parts)
            reduced_img_path[-5] = 'downsampled_files'
            reduced_img_path = Path(*reduced_img_path).with_suffix('.png')
            if reduced_img_path.is_file():
                x = str(reduced_img_path.resolve())
        elif self.__class__.__name__ in ['SIIM']:
            reduced_img_path = list(Path(x).parts)
            reduced_img_path[reduced_img_path.index('dicom-images-train')] = 'downsampled_files'
            reduced_img_path = Path(*reduced_img_path).with_suffix('.png')
            assert reduced_img_path.is_file()
            x = str(reduced_img_path.resolve())
        elif self.__class__.__name__ in ['VinDr']:
            reduced_img_path = list(Path(x).parts)
            reduced_img_path[-2] = 'downsampled_files'
            reduced_img_path = Path(*reduced_img_path).with_suffix('.png')
            assert reduced_img_path.is_file()
            x = str(reduced_img_path.resolve())
        elif self.__class__.__name__ in ['ISIC', 'ODIR']:
            reduced_img_path = list(Path(x).parts)
            reduced_img_path[-2] = 'downsampled_files'
            reduced_img_path = Path(*reduced_img_path).with_suffix('.png')
            if reduced_img_path.is_file():
                x = str(reduced_img_path.resolve())

        if self.__class__.__name__ in ['PadChest']:
            img = np.array(Image.open(x))
            img = np.uint8(img/(2**16)*255)
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
            return self.transform_(Image.fromarray(img))
        else:
            return self.transform_(Image.open(x).convert("RGB"))


class ConcatImageDataset(SubpopDataset):
    """
    Returns a single dataset from a list of BaseImageDatasets
    """
    N_STEPS = 30001
    CHECKPOINT_FREQ = 1000
    N_WORKERS = 16
    INPUT_SHAPE = (3, 224, 224,)

    def __init__(self, dss):
        self.ds = ConcatDataset(dss)
        self.x = sum([i.x for i in dss], [])
        self.y = sum([i.y for i in dss], [])
        self.a = sum([i.a for i in dss], [])
        self.idx = list(range(len(self.x)))
        self.data_type = "images"
        self._count_groups()
        super().__init__(None, None, None, None)

    def __getitem__(self, idx):
        return self.ds[idx]

    def __len__(self):
        return len(self.idx)


class SubsetImageDataset(SubpopDataset):
    """
    Subsets from an existing dataset based on provided indices
    """
    N_STEPS = 30001
    CHECKPOINT_FREQ = 1000
    N_WORKERS = 16
    INPUT_SHAPE = (3, 224, 224,)

    def __init__(self, ds, idxs):
        self.orig_ds = ds
        self.ds = Subset(ds, idxs)
        self.x = [ds.x[i] for i in idxs]
        self.a = [ds.a[i] for i in idxs]
        self.y = [ds.y[i] for i in idxs]
        self.idx = list(range(len(self.x)))
        self.data_type = "images"
        self.attr_name = ds.attr_name
        self.task_name = ds.task_name
        self._count_groups()
        super().__init__(None, None, None, None)

    def __getitem__(self, idx):
        return self.ds[idx]

    def __len__(self):
        return len(self.ds)


class MIMIC(BaseImageDataset):
    N_STEPS = 30001
    CHECKPOINT_FREQ = 1000
    N_WORKERS = 16
    INPUT_SHAPE = (3, 224, 224,)
    AVAILABLE_ATTRS = ['sex', 'age', 'ethnicity', 'sex_ethnicity']
    TASKS = [
        'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema'
    ]

    def __init__(self, data_path, split, hparams, group_def='group', subsample_type=None, duplicates=None, override_attr=None, subset_query=None):
        metadata = os.path.join(data_path, "MIMIC-CXR-JPG", 'subpop_fair_meta', "metadata.csv")
        super().__init__(metadata, split, hparams, group_def, subsample_type, duplicates, override_attr, subset_query)


class CheXpert(BaseImageDataset):
    N_STEPS = 30001
    CHECKPOINT_FREQ = 1000
    N_WORKERS = 16
    INPUT_SHAPE = (3, 224, 224,)
    AVAILABLE_ATTRS = ['sex', 'age', 'ethnicity', 'sex_ethnicity']
    TASKS = [
        'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema'
    ]

    def __init__(self, data_path, split, hparams, group_def='group', subsample_type=None, duplicates=None, override_attr=None, subset_query=None):
        metadata = os.path.join(data_path, "chexpert", 'subpop_fair_meta', "metadata.csv")
        super().__init__(metadata, split, hparams, group_def, subsample_type, duplicates, override_attr, subset_query)


class NIH(BaseImageDataset):
    N_STEPS = 30001
    CHECKPOINT_FREQ = 1000
    N_WORKERS = 16
    INPUT_SHAPE = (3, 224, 224,)
    AVAILABLE_ATTRS = ['sex', 'age']
    TASKS = [
        'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema'
    ]

    def __init__(self, data_path, split, hparams, group_def='group', subsample_type=None, duplicates=None, override_attr=None, subset_query=None):
        metadata = os.path.join(data_path, "ChestXray8", 'subpop_fair_meta', "metadata.csv")
        super().__init__(metadata, split, hparams, group_def, subsample_type, duplicates, override_attr, subset_query)


class PadChest(BaseImageDataset):
    N_STEPS = 30001
    CHECKPOINT_FREQ = 1000
    N_WORKERS = 16
    INPUT_SHAPE = (3, 224, 224,)
    AVAILABLE_ATTRS = ['sex', 'age']
    TASKS = [
        'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema'
    ]

    def __init__(self, data_path, split, hparams, group_def='group', subsample_type=None, duplicates=None, override_attr=None, subset_query=None):
        metadata = os.path.join(data_path, "PadChest", 'subpop_fair_meta', "metadata.csv")
        super().__init__(metadata, split, hparams, group_def, subsample_type, duplicates, override_attr, subset_query)


class VinDr(BaseImageDataset):
    N_STEPS = 30001
    CHECKPOINT_FREQ = 1000
    N_WORKERS = 16
    INPUT_SHAPE = (3, 224, 224,)
    AVAILABLE_ATTRS = ['sex', 'age']
    TASKS = [
        'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema'
    ]
    SPLITS = {
        'te': 2
    }

    def __init__(self, data_path, split, hparams, group_def='group', subsample_type=None, duplicates=None, override_attr=None, subset_query=None):
        metadata = os.path.join(data_path, "vindr-cxr", 'subpop_fair_meta', "metadata_eval.csv")
        super().__init__(metadata, split, hparams, group_def, subsample_type, duplicates, override_attr, subset_query)

        
class SIIM(BaseImageDataset):
    N_STEPS = 30001
    CHECKPOINT_FREQ = 1000
    N_WORKERS = 16
    INPUT_SHAPE = (3, 224, 224,)
    AVAILABLE_ATTRS = ['sex', 'age']
    TASKS = [
        'Pneumothorax'
    ]
    SPLITS = {
        'te': 2
    }

    def __init__(self, data_path, split, hparams, group_def='group', subsample_type=None, duplicates=None, override_attr=None, subset_query=None):
        metadata = os.path.join(data_path, "SIIM", 'subpop_fair_meta', "metadata.csv")
        super().__init__(metadata, split, hparams, group_def, subsample_type, duplicates, override_attr, subset_query)


class ISIC(BaseImageDataset):
    N_STEPS = 30001
    CHECKPOINT_FREQ = 1000
    N_WORKERS = 16
    INPUT_SHAPE = (3, 224, 224,)
    AVAILABLE_ATTRS = ['sex', 'age']
    TASKS = [
        'No Finding'
    ]

    def __init__(self, data_path, split, hparams, group_def='group', subsample_type=None, duplicates=None, override_attr=None, subset_query=None):
        metadata = os.path.join(data_path, "ISIC", 'subpop_fair_meta', "metadata.csv")
        super().__init__(metadata, split, hparams, group_def, subsample_type, duplicates, override_attr, subset_query)


class ODIR(BaseImageDataset):
    N_STEPS = 30001
    CHECKPOINT_FREQ = 1000
    N_WORKERS = 16
    INPUT_SHAPE = (3, 224, 224,)
    AVAILABLE_ATTRS = ['sex', 'age']
    TASKS = [
        'No Finding', 'Cataract', 'Retinopathy'
    ]

    def __init__(self, data_path, split, hparams, group_def='group', subsample_type=None, duplicates=None, override_attr=None, subset_query=None):
        metadata = os.path.join(data_path, "ODIR", 'subpop_fair_meta', "metadata.csv")
        super().__init__(metadata, split, hparams, group_def, subsample_type, duplicates, override_attr, subset_query)
