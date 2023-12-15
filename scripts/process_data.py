import argparse
import os
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

TASKS = [
    'No Finding', 'Atelectasis', 'Cardiomegaly',  'Effusion',  'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema'
]


def bin_age(x):
    if pd.isnull(x): return None
    elif 0 <= x < 18: return 4
    elif 18 <= x < 40: return 3
    elif 40 <= x < 60: return 2
    elif 60 <= x < 80: return 1
    else: return 0


def generate_metadata(data_path, datasets):
    dataset_metadata_generators = {
        'mimic': generate_metadata_mimic_cxr,
        'chexpert': generate_metadata_chexpert,
        'nih': generate_metadata_nih,
        'padchest': generate_metadata_padchest,
        'vindr': generate_metadata_vindr,
        'siim': generate_metadata_siim,
        'isic': generate_metadata_isic,
        'odir': generate_metadata_odir
    }
    for dataset in datasets:
        dataset_metadata_generators[dataset](data_path)


def generate_metadata_mimic_cxr(data_path, test_pct=0.15, val_pct=0.15):
    logging.info("Generating metadata for MIMIC-CXR...")
    img_dir = Path(os.path.join(data_path, "MIMIC-CXR-JPG"))

    assert (img_dir/'mimic-cxr-2.0.0-metadata.csv.gz').is_file()
    assert (img_dir/'patients.csv.gz').is_file(), \
        'Please download patients.csv.gz and admissions.csv.gz from MIMIC-IV and place it in the image folder.'
    assert (img_dir/'files/p19/p19316207/s55102753/31ec769b-463d6f30-a56a7e09-76716ec1-91ad34b6.jpg').is_file()

    def ethnicity_mapping(x):
        if pd.isnull(x):
            return 3
        elif x.startswith("WHITE"):
            return 0
        elif x.startswith("BLACK"):
            return 1
        elif x.startswith("ASIAN"):
            return 2
        return 3

    patients = pd.read_csv(img_dir/'patients.csv.gz')
    ethnicities = pd.read_csv(img_dir/'admissions.csv.gz').drop_duplicates(
        subset=['subject_id']).set_index('subject_id')['race'].to_dict()
    patients['ethnicity'] = patients['subject_id'].map(ethnicities).map(ethnicity_mapping)
    labels = pd.read_csv(img_dir/'mimic-cxr-2.0.0-negbio.csv.gz')
    meta = pd.read_csv(img_dir/'mimic-cxr-2.0.0-metadata.csv.gz')

    df = meta.merge(patients, on='subject_id').merge(labels, on=['subject_id', 'study_id'])
    df['age_decile'] = pd.cut(df['anchor_age'], bins=list(range(0, 101, 10))).apply(lambda x: f'{x.left}-{x.right}').astype(str)
    df['age'] = df['anchor_age'].apply(bin_age)
    df['frontal'] = df.ViewPosition.isin(['AP', 'PA'])

    df['filename'] = df.apply(
        lambda x: os.path.join(
            img_dir,
            'files', f'p{str(x["subject_id"])[:2]}', f'p{x["subject_id"]}', f's{x["study_id"]}', f'{x["dicom_id"]}.jpg'
        ), axis=1)
    df = df[df.anchor_age > 0]

    attr_mapping = {'M_0': 0, 'F_0': 1, 'M_1': 2, 'F_1': 3, 'M_2': 4, 'F_2': 5, 'M_3': 6, 'F_3': 7}
    df['sex_ethnicity'] = (df['gender'] + '_' + df['ethnicity'].astype(str)).map(attr_mapping)
    df['gender'] = (df['gender'] == 'M').astype(int)

    df = df.rename(columns={
        'gender': 'sex',
        'Pleural Effusion': 'Effusion',
        'Lung Opacity': 'Airspace Opacity'
    })    

    for t in TASKS:
        # treat uncertain labels as negative
        df[t] = (df[t].fillna(0.0) == 1.0).astype(int)

    train_val_idx, test_idx = train_test_split(df.index, test_size=test_pct, random_state=42)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_pct/(1-test_pct), random_state=42)

    df['split'] = 0
    df.loc[val_idx, 'split'] = 1
    df.loc[test_idx, 'split'] = 2

    (img_dir/'subpop_fair_meta').mkdir(exist_ok=True)
    df.to_csv(os.path.join(img_dir, 'subpop_fair_meta', "metadata.csv"), index=False)


def generate_metadata_chexpert(data_path, test_pct=0.15, val_pct=0.15):
    logging.info("Generating metadata for CheXpert...")
    chexpert_dir = Path(os.path.join(data_path, "chexpert"))
    assert (chexpert_dir/'train.csv').is_file()
    assert (chexpert_dir/'train/patient48822/study1/view1_frontal.jpg').is_file()
    assert (chexpert_dir/'valid/patient64636/study1/view1_frontal.jpg').is_file()
    assert (chexpert_dir/'CHEXPERT DEMO.xlsx').is_file()

    df = pd.concat([pd.read_csv(chexpert_dir/'train.csv'), pd.read_csv(chexpert_dir/'valid.csv')], ignore_index=True)

    df['filename'] = df['Path'].astype(str).apply(lambda x: os.path.join(chexpert_dir, x[x.index('/')+1:]))
    df['subject_id'] = df['Path'].apply(lambda x: int(Path(x).parent.parent.name[7:])).astype(str)
    df = df[df.Sex.isin(['Male', 'Female'])]
    details = pd.read_excel(chexpert_dir/'CHEXPERT DEMO.xlsx', engine='openpyxl')[['PATIENT', 'PRIMARY_RACE']]
    details['subject_id'] = details['PATIENT'].apply(lambda x: x[7:]).astype(int).astype(str)

    df = pd.merge(df, details, on='subject_id', how='inner').reset_index(drop=True)

    def cat_race(r):
        if isinstance(r, str):
            if r.startswith('White'):
                return 0
            elif r.startswith('Black'):
                return 1
            elif r.startswith('Asian'):
                return 2
        return 3

    df['ethnicity'] = df['PRIMARY_RACE'].apply(cat_race)
    attr_mapping = {'Male_0': 0, 'Female_0': 1, 'Male_1': 2, 'Female_1': 3, 'Male_2': 4, 'Female_2': 5, 'Male_3': 6, 'Female_3': 7}
    df['sex_ethnicity'] = (df['Sex'] + '_' + df['ethnicity'].astype(str)).map(attr_mapping)
    df['age'] = df['Age'].apply(bin_age)
    df = df.rename(columns={
        'Sex': 'sex',
        'Pleural Effusion': 'Effusion',
        'Lung Opacity': 'Airspace Opacity',
    })    
    df['sex'] = (df['sex'] == 'Male').astype(int)

    for t in TASKS:
        # treat uncertain labels as negative
        df[t] = (df[t].fillna(0.0) == 1.0).astype(int)

    df = df[df.Age > 0]

    train_val_idx, test_idx = train_test_split(df.index, test_size=test_pct, random_state=42)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_pct/(1-test_pct), random_state=42)

    df['split'] = 0
    df.loc[val_idx, 'split'] = 1
    df.loc[test_idx, 'split'] = 2

    (chexpert_dir/'subpop_fair_meta').mkdir(exist_ok=True)
    df.to_csv(os.path.join(chexpert_dir, 'subpop_fair_meta', "metadata.csv"), index=False)


def generate_metadata_nih(data_path, test_pct=0.15, val_pct=0.15):
    logging.info("Generating metadata for ChestX-ray8...")
    nih_dir = Path(os.path.join(data_path, "ChestXray8"))
    assert (nih_dir/'images'/'00002072_003.png').is_file()

    df = pd.read_csv(nih_dir/"Data_Entry_2017_v2020.csv")
    df['labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))

    for label in TASKS:
        df[label] = (df['labels'].apply(lambda x: label in x)).astype(int)

    df['age'] = df['Patient Age'].apply(bin_age)
    df['sex'] = (df['Patient Gender'] == 'M').astype(int)
    df['frontal'] = True

    df['filename'] = df['Image Index'].astype(str).apply(lambda x: os.path.join(nih_dir, 'images', x))

    train_val_idx, test_idx = train_test_split(df.index, test_size=test_pct, random_state=42)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_pct/(1-test_pct), random_state=42)

    df['split'] = 0
    df.loc[val_idx, 'split'] = 1
    df.loc[test_idx, 'split'] = 2

    (nih_dir/'subpop_fair_meta').mkdir(exist_ok=True)
    df.to_csv(os.path.join(nih_dir, 'subpop_fair_meta', "metadata.csv"), index=False)


def generate_metadata_padchest(data_path, test_pct=0.15, val_pct=0.15):
    logging.info("Generating metadata for PadChest... This might take a few minutes.")
    pc_dir = Path(os.path.join(data_path, "PadChest"))
    assert (pc_dir/'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv').is_file()
    assert (pc_dir/'images-224'/'304569295539964045020899697180335460865_r2fjf7.png').is_file()

    df = pd.read_csv(pc_dir/'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv')
    df = df[['ImageID', 'StudyID', 'PatientID', 'PatientBirth', 'PatientSex_DICOM', 'ViewPosition_DICOM', 'Projection', 'Labels']]
    df = df[~df["Labels"].isnull()]
    df['filename'] = df['ImageID'].astype(str).apply(lambda x: os.path.join(pc_dir, 'images-224', x))
    df = df[df["filename"].apply(lambda x: os.path.exists(x))]
    df = df[df.Projection.isin(['PA', 'L', 'AP_horizontal', 'AP'])]
    df['Age'] = 2017 - df['PatientBirth']
    df = df[~df['Age'].isnull()]

    df['frontal'] = ~(df['Projection'] == 'L')
    df = df[~df['Labels'].apply(lambda x: 'exclude' in x or 'unchanged' in x)]

    mapping = dict()
    mapping["Consolidation"] = ["air bronchogram"]
    mapping['No Finding'] = ['normal']

    for pathology in TASKS:
        mask = df["Labels"].str.contains(pathology.lower())
        if pathology in mapping:
            for syn in mapping[pathology]:
                mask |= df["Labels"].str.contains(syn.lower())
        df[pathology] = mask.astype(int)

    df['age'] = df['Age'].apply(bin_age)
    df['sex'] = (df['PatientSex_DICOM'] == 'M').astype(int)

    train_val_idx, test_idx = train_test_split(df.index, test_size=test_pct, random_state=42)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_pct/(1-test_pct), random_state=42)

    df['split'] = 0
    df.loc[val_idx, 'split'] = 1
    df.loc[test_idx, 'split'] = 2

    (pc_dir/'subpop_fair_meta').mkdir(exist_ok=True)
    df.to_csv(os.path.join(pc_dir, 'subpop_fair_meta', "metadata.csv"), index=False)


def generate_metadata_vindr(data_path, val_pct=0.15):
    logging.info("Generating metadata for VinDr-CXR... This will take 30+ minutes.")
    vin_dir = Path(os.path.join(data_path, "vindr-cxr"))

    assert (vin_dir/'train'/'d23be2fc84b61c8250b0047619c34f1a.dicom').is_file()
    train_df = pd.read_csv(vin_dir/'annotations'/'image_labels_train.csv')
    test_df = pd.read_csv(vin_dir/'annotations'/'image_labels_test.csv')

    train_df['filename'] = train_df['image_id'].astype(str).apply(lambda x: os.path.join(vin_dir, 'train', x+'.dicom'))
    train_df['split'] = 0
    train_idx, val_idx = train_test_split(train_df.index, test_size=val_pct, random_state=42)
    train_df.loc[val_idx, 'split'] = 1
    test_df['filename'] = test_df['image_id'].astype(str).apply(lambda x: os.path.join(vin_dir, 'test', x+'.dicom'))
    test_df['split'] = 2
    df = pd.concat([train_df, test_df], ignore_index=True)

    diseases = [i for i in df.columns if i not in ['image_id', 'rad_id', 'filename', 'split', 'Other diseases', 'Other disease']]
    df = df.pivot_table(values=diseases, index=['image_id', 'filename', 'split'], columns='rad_id')
    df.columns = [i + f'_{j}' for (i, j) in df.columns]

    for t in diseases:
        df[t] = df[[t + f'_R{i}'for i in range(1, 18)]].mode(axis=1)[0]

    df = df.reset_index().rename(columns={
        'Pleural effusion': 'Effusion',
        'No finding': 'No Finding'
    })

    df['exists'] = df['filename'].apply(lambda x: Path(x).is_file())

    import pydicom
    df['sex_char'] = None
    df['age_yr'] = None
    for idx, row in tqdm(df.iterrows()):
        dicom_obj = pydicom.filereader.dcmread(row['filename'])
        df.loc[idx, 'sex_char'] = dicom_obj[0x0010, 0x0040].value
        try:
            df.loc[idx, 'age_yr'] = int(dicom_obj[0x0010, 0x1010].value[:-1])
        except:
            # no age
            pass

    # 13k patients have missing age
    df['sex'] = df['sex_char'].map({'M': 1, 'F': 0}, na_action=None)
    df['age'] = df['age_yr'].apply(bin_age)
    df['frontal'] = True

    df['to_take_eval'] = (~df['sex'].isnull()) & (~df['age'].isnull())
    eval_df = df.loc[df['to_take_eval']]
    eval_df['split'] = 2
    eval_df['sex'] = eval_df['sex'].astype(int)
    eval_df['age'] = eval_df['age'].astype(int)

    (vin_dir/'subpop_fair_meta').mkdir(exist_ok=True)

    df.to_csv(os.path.join(vin_dir, 'subpop_fair_meta', "metadata.csv"), index=False)
    eval_df.to_csv(os.path.join(vin_dir, 'subpop_fair_meta', "metadata_eval.csv"), index=False)


def generate_metadata_siim(data_path, *args, **kwargs):
    logging.info("Generating metadata for SIIM... This will take 30+ minutes.")
    siim_dir = Path(os.path.join(data_path, "SIIM"))

    assert (siim_dir/'dicom-images-train'/'1.2.276.0.7230010.3.1.2.8323329.11233.1517875231.878551'/'1.2.276.0.7230010.3.1.3.8323329.11233.1517875231.878550'/'1.2.276.0.7230010.3.1.4.8323329.11233.1517875231.878552.dcm').is_file()
    df = pd.read_csv(siim_dir/'train-rle.csv').set_index('ImageId')
    df['Pneumothorax'] = ~(df[' EncodedPixels'] == ' -1')

    import pydicom
    df['sex_char'] = None
    df['age_yr'] = None
    df['filename'] = None
    for i in tqdm((siim_dir/'dicom-images-train').glob('**/*.dcm')):
        dicom_obj = pydicom.filereader.dcmread(i)
        idx = i.name[:-4]
        if idx not in df.index:
            continue
        df.loc[idx, 'filename'] = str(i)
        df.loc[idx, 'sex_char'] = dicom_obj[0x0010, 0x0040].value
        try:
            df.loc[idx, 'age_yr'] = int(dicom_obj[0x0010, 0x1010].value)
        except:
            # no age
            pass

    df = df[~df.filename.isnull()]
    df['sex'] = df['sex_char'].map({'M': 1, 'F': 0}, na_action=None)
    df['age'] = df['age_yr'].apply(bin_age)
    df['frontal'] = True
    print(f"Before dropping missing age/sex: {len(df)}")
    df = df[(~df['sex'].isnull()) & (~df['age'].isnull())]
    print(f"After dropping missing age/sex: {len(df)}")
    df['split'] = 2
    df['sex'] = df['sex'].astype(int)
    df['age'] = df['age'].astype(int)

    (siim_dir/'subpop_fair_meta').mkdir(exist_ok=True)

    df.to_csv(os.path.join(siim_dir, 'subpop_fair_meta', "metadata.csv"), index=False)


def generate_metadata_isic(data_path, test_pct=0.2, val_pct=0.15):
    logging.info("Generating metadata for ISIC...")
    isic_dir = Path(os.path.join(data_path, "ISIC"))
    assert (isic_dir/'ISIC_2020_Training_GroundTruth_v2.csv').is_file()
    assert (isic_dir/'train'/'ISIC_3707185.jpg').is_file()
    df = pd.read_csv(isic_dir/'ISIC_2020_Training_GroundTruth_v2.csv').dropna(subset=['sex', 'age_approx'])
    df['filename'] = df['image_name'].astype(str).apply(lambda x: os.path.join(isic_dir, 'train', x+'.jpg'))
    df['No Finding'] = (df['benign_malignant'] == 'benign').astype(int)
    df['sex'] = (df['sex'] == 'male').astype(int)
    df['age'] = df['age_approx'].apply(bin_age)

    df = df.set_index('image_name')
    train_val_idx, test_idx = train_test_split(df.index, test_size=test_pct, random_state=42)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_pct/(1-test_pct), random_state=42)

    df['split'] = 0
    df.loc[val_idx, 'split'] = 1
    df.loc[test_idx, 'split'] = 2

    (isic_dir/'subpop_fair_meta').mkdir(exist_ok=True)
    df.to_csv(os.path.join(isic_dir, 'subpop_fair_meta', "metadata.csv"), index=False)


def generate_metadata_odir(data_path, test_pct=0.2, val_pct=0.15):
    logging.info("Generating metadata for ODIR...")
    odir_dir = Path(os.path.join(data_path, "ODIR"))
    assert (odir_dir/'ODIR-5K_Training_Annotations(Updated)_V2.xlsx').is_file()
    assert (odir_dir/'ODIR-5K_Training_Dataset'/'4202_left.jpg').is_file()
    df = pd.read_excel(odir_dir/'ODIR-5K_Training_Annotations(Updated)_V2.xlsx').dropna(subset=['Patient Age', 'Patient Sex'])
    df = pd.concat([
        df[['ID', 'Patient Age', 'Patient Sex', 'Left-Fundus', 'Left-Diagnostic Keywords']].rename(columns={
            'Left-Fundus': 'file', 'Left-Diagnostic Keywords': 'keywords'}),
        df[['ID', 'Patient Age', 'Patient Sex', 'Right-Fundus', 'Right-Diagnostic Keywords']].rename(columns={
            'Right-Fundus': 'file', 'Right-Diagnostic Keywords': 'keywords'}),
    ], ignore_index=True)
    df['No Finding'] = df.keywords.str.lower().str.contains('normal fundus').astype(int)
    df['Cataract'] = df.keywords.str.lower().str.contains('cataract').astype(int)
    df['Retinopathy'] = df.keywords.str.lower().str.contains('retinopathy').astype(int)
    df['sex'] = (df['Patient Sex'] == 'Male').astype(int)
    df['age'] = df['Patient Age'].apply(bin_age)
    df = df[~(df.age == 4)]  # drop patients 0-18 since there are too few samples
    df['filename'] = df['file'].astype(str).apply(lambda x: os.path.join(odir_dir, 'ODIR-5K_Training_Dataset', x))
    all_ids = df.ID.unique()
    train_val_idx, test_idx = train_test_split(all_ids, test_size=test_pct, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_pct/(1-test_pct), random_state=42)

    df['split'] = 0
    df.loc[df.ID.isin(val_idx), 'split'] = 1
    df.loc[df.ID.isin(test_idx), 'split'] = 2

    (odir_dir/'subpop_fair_meta').mkdir(exist_ok=True)
    df.to_csv(os.path.join(odir_dir, 'subpop_fair_meta', "metadata.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset')
    parser.add_argument('--dataset', nargs='+', type=str, required=False,
                        default=['mimic', 'chexpert', 'nih', 'padchest', 'vindr', 'siim', 'isic', 'odir'])
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()

    generate_metadata(args.data_path, args.dataset)
