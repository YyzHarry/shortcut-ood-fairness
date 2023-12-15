# Downloading Datasets

## MIMIC-CXR
1. [Obtain access](https://mimic-cxr.mit.edu/about/access/) to the MIMIC-CXR-JPG Database on PhysioNet and download the [dataset](https://physionet.org/content/mimic-cxr-jpg/2.0.0/). We recommend downloading from the GCP bucket:

```bash
gcloud auth login
mkdir MIMIC-CXR-JPG
gsutil -m rsync -d -r gs://mimic-cxr-jpg-2.0.0.physionet.org MIMIC-CXR-JPG
```

2. In order to obtain demographic information for each patient, you will need to obtain access to [MIMIC-IV](https://physionet.org/content/mimiciv/). Download `core/patients.csv.gz` and `core/admissions.csv.gz` and place the files in the `MIMIC-CXR-JPG` directory.

3. Move or create a symbolic link to the `MIMIC-CXR-JPG` folder from your data directory.

4. Run `python -m scripts.process_data mimic_cxr --data_path <data_path>`.

## CheXpert

1. Download the [downsampled CheXpert dataset](http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip) and extract it.

2. Register for an account and download the CheXpert demographics data [here](https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf). Place the `CHEXPERT DEMO.xlsx` in your CheXpert directory. 

3. Move or create a symbolic link to the `CheXpert-v1.0-small` folder named `chexpert` in your data directory.

4. Run `python -m scripts.process_data chexpert --data_path <data_path>`.


## ChestX-ray8 (NIH)

1. Download the `images` folder and the `Data_Entry_2017_v2020.csv` file from [this link](https://nihcc.app.box.com/v/ChestXray-NIHCC). Move the csv file into the parent directory of the `images` folder.

2. Move or create a symbolic link to the parent folder named `ChestXray8` in your data directory.

3. Run `python -m scripts.process_data nih --data_path <data_path>`.

## PadChest

1. We use a resized version of PadChest, which can be downloaded [here](https://academictorrents.com/details/96ebb4f92b85929eadfb16761f310a6d04105797).

2. Unzip `images-224.tar`.

3. Move or create a symbolic link to this folder named `PadChest` in your data directory. This directory should contain the folder `images-224` and the file `PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv`.

4. Run `python -m scripts.process_data padchest --data_path <data_path>`.

## VinDr-CXR

1. [Obtain access](https://mimic-cxr.mit.edu/about/access/) to the VinDr-CXR dataset on PhysioNet and download the [dataset](https://physionet.org/content/vindr-cxr/1.0.0/). 

2. Move or create a symbolic link to this folder named `vindr-cxr` in your data directory.

3. Run `python -m scripts.process_data vindr --data_path <data_path>`.

## SIIM

1. Download the SIIM dataset from [Kaggle](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation).
```
kaggle datasets download -d jesperdramsch/siim-acr-pneumothorax-segmentation-data
```

2.  Move or create a symbolic link to this folder named `SIIM` in your data directory.

3. Run `python -m scripts.process_data siim --data_path <data_path>`.

## ISIC

1. Download the [ISIC 2020 Challenge dataset](https://challenge2020.isic-archive.com/) (the JPEG zip file and the metadata v2 file). Unzip the zip file.

2. Move or create a symbolic link to the parent folder named `ISIC` in your data directory. This folder should contain the file `ISIC_2020_Training_GroundTruth_v2.csv` and the folder `train`.

3. Run `python -m scripts.process_data isic --data_path <data_path>`.


## ODIR

1. Download the [ODIR dataset](https://academictorrents.com/details/cf3b8d5ecdd4284eb9b3a80fcfe9b1d621548f72). Unzip the training images.

2. Move or create a symbolic link to the parent folder named `ODIR` in your data directory.

3. Run `python -m scripts.process_data odir --data_path <data_path>`.