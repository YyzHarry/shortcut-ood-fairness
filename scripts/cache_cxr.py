import os
import argparse
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut


class CXRDataset(Dataset):

    def __init__(self, df, min_dim, dataset, overwrite=False):
        super().__init__()
        self.df = df
        self.overwrite = overwrite
        self.transform = transforms.Resize(min_dim)
        self.dataset = dataset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = df.iloc[idx]
        img_path = Path(row['filename'])
        reduced_img_path = list(img_path.parts)
        if self.dataset == 'mimic':
            assert reduced_img_path[-5] == 'files', reduced_img_path
            reduced_img_path[-5] = 'downsampled_files'
        elif self.dataset == 'vindr':
            assert reduced_img_path[-2] in ('train', 'test'), reduced_img_path
            reduced_img_path[-2] = 'downsampled_files'
        elif self.dataset == 'siim':
            assert 'dicom-images-train' in reduced_img_path, reduced_img_path
            reduced_img_path[reduced_img_path.index('dicom-images-train')] = 'downsampled_files'
        elif self.dataset == 'isic':
            assert reduced_img_path[-2] == 'train', reduced_img_path
            reduced_img_path[-2] = 'downsampled_files'
        elif self.dataset == 'odir':
            assert reduced_img_path[-2] == 'ODIR-5K_Training_Dataset', reduced_img_path
            reduced_img_path[-2] = 'downsampled_files'
        reduced_img_path = Path(*reduced_img_path).with_suffix('.png')

        if self.overwrite or not reduced_img_path.is_file():
            reduced_img_path.parent.mkdir(exist_ok=True, parents=True)

            if self.dataset in ['mimic', 'isic', 'odir']:
                img = Image.open(img_path).convert("RGB")
            elif self.dataset in ['vindr', 'siim']:
                dicom_obj = pydicom.filereader.dcmread(img_path)
                img = apply_modality_lut(dicom_obj.pixel_array, dicom_obj)
                img = pydicom.pixel_data_handlers.apply_windowing(img, dicom_obj)

                # Photometric Interpretation to see if the image needs to be inverted
                mode = dicom_obj[0x28, 0x04].value
                bitdepth = dicom_obj[0x28, 0x101].value
                if img.max() < 256:
                    bitdepth = 8

                if mode == "MONOCHROME1":
                    img = -1 * img + 2**float(bitdepth)
                elif mode == "MONOCHROME2":
                    pass
                else:
                    raise Exception("Unknown Photometric Interpretation mode")

                img = img / img.max()
                img = Image.fromarray(np.uint8(img * 255)).convert('RGB')

            new_img = self.transform(img)
            new_img.save(reduced_img_path)
            return 1
        else:
            return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downsample CXR datasets')
    parser.add_argument('--dataset', type=str, choices=['mimic', 'vindr', 'siim', 'isic', 'odir'])
    parser.add_argument('--data_path', type=str, default='/data/netmit/rf-sleep/yuzhe/subpop-fairness/data')
    parser.add_argument('--n_workers', type=int, default=64)
    parser.add_argument('--min_dim', type=int, default=256)
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing cached files')
    args = parser.parse_args()

    if args.dataset == 'mimic':
        mimic_dir = Path(os.path.join(args.data_path, "MIMIC-CXR-JPG"))
        assert (mimic_dir/'files/p19/p19316207/s55102753/31ec769b-463d6f30-a56a7e09-76716ec1-91ad34b6.jpg').is_file()
        assert (mimic_dir/'subpop_fair_meta'/'metadata.csv').is_file(), \
            "Please run download.py to generate the metadata for `mimic_cxr` first!"
        df = pd.read_csv(mimic_dir/'subpop_fair_meta'/'metadata.csv')
    elif args.dataset == 'vindr':
        vin_dir = Path(os.path.join(args.data_path, "vindr-cxr"))
        assert (vin_dir/'subpop_fair_meta'/'metadata.csv').is_file(), \
            "Please run download.py to generate the metadata for `vindr` first!"
        df = pd.read_csv(vin_dir/'subpop_fair_meta'/'metadata.csv')
    elif args.dataset == 'siim':
        siim_dir = Path(os.path.join(args.data_path, "SIIM"))
        assert (siim_dir/'subpop_fair_meta'/'metadata.csv').is_file(), \
            "Please run download.py to generate the metadata for `siim` first!"
        df = pd.read_csv(siim_dir/'subpop_fair_meta'/'metadata.csv')
    elif args.dataset == 'isic':
        isic_dir = Path(os.path.join(args.data_path, "ISIC"))
        assert (isic_dir/'subpop_fair_meta'/'metadata.csv').is_file(), \
            "Please run download.py to generate the metadata for `isic` first!"
        df = pd.read_csv(isic_dir/'subpop_fair_meta'/'metadata.csv')
    elif args.dataset == 'odir':
        odir_dir = Path(os.path.join(args.data_path, "ODIR"))
        assert (odir_dir/'subpop_fair_meta'/'metadata.csv').is_file(), \
            "Please run download.py to generate the metadata for `odir` first!"
        df = pd.read_csv(odir_dir/'subpop_fair_meta'/'metadata.csv')
    else:
        raise NotImplementedError(f"{args.dataset} not supported.")

    ds = CXRDataset(df, args.min_dim, args.dataset, args.overwrite)
    dl = DataLoader(ds, batch_size=64, num_workers=args.n_workers, shuffle=False)

    for i in tqdm(dl):
        pass
