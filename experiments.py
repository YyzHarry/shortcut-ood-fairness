import os
import json
from pathlib import Path
from itertools import product


def combinations_base(grid):
    return list(dict(zip(grid.keys(), values)) for values in product(*grid.values()))


def combinations(grid):
    sub_exp_names = set()
    args = []
    for i in grid:
        if isinstance(grid[i], dict):
            for j in grid[i]:
                sub_exp_names.add(j)
    if len(sub_exp_names) == 0:
        return combinations_base(grid)

    for i in grid:
        if isinstance(grid[i], dict):
            assert set(list(grid[i].keys())) == sub_exp_names, f'{i} does not have all sub exps ({sub_exp_names})'
    for n in sub_exp_names:
        sub_grid = grid.copy()
        for i in sub_grid:
            if isinstance(sub_grid[i], dict):
                sub_grid[i] = sub_grid[i][n]
        args += combinations_base(sub_grid)
    return args


def get_hparams(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment]().get_hparams()


def get_script_name(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment].fname


def match_two_stage(hparams, folder):
    bank = {}
    final_list = []
    for i in Path(folder).glob('**/done'):
        args = json.load((i.parent/'args.json').open('r'))
        if args['algorithm'] == 'ERM':
            bank[i] = args

    for hparam in hparams:
        for b, c in bank.items():
            for j in hparam:
                # skip "attr" checks as ERM only has runs with "attr=sex"
                if j not in ['algorithm', 'attr'] and hparam[j] != c[j]:
                    break
            else:
                hparam['stage1_folder'] = str(b.parent)
                final_list.append(hparam)
                break
        else:
            print(f"Corresponding stage 1 folder not found for {hparam}")
    return final_list


# ========== evaluation ========== #
class eval_race_mimic:
    fname = 'eval'
    root_dir = Path('/path/to/output/grid_race_mimic')

    def __init__(self):
        dirs = []
        for i in self.root_dir.glob('**/done'):
            if not (i.parent/'done_eval').is_file():
                dirs.append(str(i.parent))

        self.hparams = {
            'model_dir': dirs
        }

    def get_hparams(self):
        return combinations(self.hparams)


# ========== training ========== #
class grid_race_mimic:
    fname = 'train'

    def __init__(self):
        self.hparams = {
            'dataset': ['MIMIC'],
            'algorithm': ['ERM', 'MA', 'GroupDRO', 'ReSample', 'DANN', 'CDANN'],
            'task': ['No Finding', 'Cardiomegaly', 'Effusion', 'Pneumothorax'],
            'attr': ['ethnicity'],
            'hparams_seed': list(range(12)),
            'group_def': ['group'],
            'use_es': [True],
            'es_metric': ['overall:AUROC'],
            'seed': [0, 1, 2], 
        }

    def get_hparams(self):
        return combinations(self.hparams)


class grid_race_chexpert:
    fname = 'train'

    def __init__(self):
        self.hparams = {
            'dataset': ['CheXpert'],
            'algorithm': ['ERM', 'MA', 'GroupDRO', 'ReSample', 'DANN', 'CDANN'],
            'task': ['No Finding', 'Cardiomegaly', 'Effusion', 'Pneumothorax'],
            'attr': ['ethnicity'],
            'hparams_seed': list(range(12)),
            'group_def': ['group'],
            'use_es': [True],
            'es_metric': ['overall:AUROC'],
            'seed': [0, 1, 2],
        }

    def get_hparams(self):
        return combinations(self.hparams)


class grid_sex_mimic:
    fname = 'train'

    def __init__(self):
        self.hparams = {
            'dataset': ['MIMIC'],
            'algorithm': ['ERM', 'MA', 'GroupDRO', 'ReSample', 'DANN', 'CDANN'],
            'task': ['No Finding', 'Cardiomegaly', 'Effusion', 'Pneumothorax'],
            'attr': ['sex'],
            'hparams_seed': list(range(12)),
            'group_def': ['group'],
            'use_es': [True],
            'es_metric': ['overall:AUROC'],
            'seed': [0, 1, 2],
        }

    def get_hparams(self):
        return combinations(self.hparams)


class grid_sex_chexpert:
    fname = 'train'

    def __init__(self):
        self.hparams = {
            'dataset': ['CheXpert'],
            'algorithm': ['ERM', 'MA', 'GroupDRO', 'ReSample', 'DANN', 'CDANN'],
            'task': ['No Finding', 'Cardiomegaly', 'Effusion', 'Pneumothorax'],
            'attr': ['sex'],
            'hparams_seed': list(range(12)),
            'group_def': ['group'],
            'use_es': [True],
            'es_metric': ['overall:AUROC'],
            'seed': [0, 1, 2],
        }

    def get_hparams(self):
        return combinations(self.hparams)


class grid_age_mimic:
    fname = 'train'

    def __init__(self):
        self.hparams = {
            'dataset': ['MIMIC'],
            'algorithm': ['ERM', 'MA', 'GroupDRO', 'ReSample', 'DANN', 'CDANN'],
            'task': ['No Finding', 'Cardiomegaly', 'Effusion', 'Pneumothorax'],
            'attr': ['age'],
            'hparams_seed': list(range(12)),
            'group_def': ['group'],
            'use_es': [True],
            'es_metric': ['overall:AUROC'],
            'seed': [0, 1, 2],
        }

    def get_hparams(self):
        return combinations(self.hparams)


class grid_age_chexpert:
    fname = 'train'

    def __init__(self):
        self.hparams = {
            'dataset': ['CheXpert'],
            'algorithm': ['ERM', 'MA', 'GroupDRO', 'ReSample', 'DANN', 'CDANN'],
            'task': ['No Finding', 'Cardiomegaly', 'Effusion', 'Pneumothorax'],
            'attr': ['age'],
            'hparams_seed': list(range(12)),
            'group_def': ['group'],
            'use_es': [True],
            'es_metric': ['overall:AUROC'],
            'seed': [0, 1, 2],
        }

    def get_hparams(self):
        return combinations(self.hparams)


class grid_sex_race_mimic:
    fname = 'train'

    def __init__(self):
        self.hparams = {
            'dataset': ['MIMIC'],
            'algorithm': ['ERM', 'MA', 'GroupDRO', 'ReSample', 'DANN', 'CDANN'],
            'task': ['No Finding', 'Cardiomegaly', 'Effusion', 'Pneumothorax'],
            'attr': ['sex_ethnicity'],
            'hparams_seed': list(range(12)),
            'group_def': ['group'],
            'use_es': [True],
            'es_metric': ['overall:AUROC'],
            'seed': [0, 1, 2],
        }

    def get_hparams(self):
        return combinations(self.hparams)


class grid_sex_race_chexpert:
    fname = 'train'

    def __init__(self):
        self.hparams = {
            'dataset': ['CheXpert'],
            'algorithm': ['ERM', 'MA', 'GroupDRO', 'ReSample', 'DANN', 'CDANN'],
            'task': ['No Finding', 'Cardiomegaly', 'Effusion', 'Pneumothorax'],
            'attr': ['sex_ethnicity'],
            'hparams_seed': list(range(12)),
            'group_def': ['group'],
            'use_es': [True],
            'es_metric': ['overall:AUROC'],
            'seed': [0, 1, 2],
        }

    def get_hparams(self):
        return combinations(self.hparams)


class grid_sex_isic:
    fname = 'train'

    def __init__(self):
        self.hparams = {
            'dataset': ['ISIC'],
            'algorithm': ['ERM', 'MA', 'GroupDRO', 'ReSample', 'DANN', 'CDANN'],
            'task': ['No Finding'],
            'attr': ['sex'],
            'hparams_seed': list(range(12)),
            'group_def': ['group'],
            'use_es': [True],
            'es_metric': ['overall:AUROC'],
            'seed': [0, 1, 2],
        }

    def get_hparams(self):
        return combinations(self.hparams)


class grid_age_isic:
    fname = 'train'

    def __init__(self):
        self.hparams = {
            'dataset': ['ISIC'],
            'algorithm': ['ERM', 'MA', 'GroupDRO', 'ReSample', 'DANN', 'CDANN'],
            'task': ['No Finding'],
            'attr': ['age'],
            'hparams_seed': list(range(12)),
            'group_def': ['group'],
            'use_es': [True],
            'es_metric': ['overall:AUROC'],
            'seed': [0, 1, 2],
        }

    def get_hparams(self):
        return combinations(self.hparams)


class grid_sex_odir:
    fname = 'train'

    def __init__(self):
        self.hparams = {
            'dataset': ['ODIR'],
            'algorithm': ['ERM', 'MA', 'GroupDRO', 'ReSample', 'DANN', 'CDANN'],
            'task': ['Retinopathy'],
            'attr': ['sex'],
            'hparams_seed': list(range(12)),
            'group_def': ['group'],
            'use_es': [True],
            'es_metric': ['overall:AUROC'],
            'seed': [0, 1, 2],
        }

    def get_hparams(self):
        return combinations(self.hparams)


class grid_age_odir:
    fname = 'train'

    def __init__(self):
        self.hparams = {
            'dataset': ['ODIR'],
            'algorithm': ['ERM', 'MA', 'GroupDRO', 'ReSample', 'DANN', 'CDANN'],
            'task': ['Retinopathy'],
            'attr': ['age'],
            'hparams_seed': list(range(12)),
            'group_def': ['group'],
            'use_es': [True],
            'es_metric': ['overall:AUROC'],
            'seed': [0, 1, 2],
        }

    def get_hparams(self):
        return combinations(self.hparams)
