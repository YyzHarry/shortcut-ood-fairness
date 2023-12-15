import argparse
import json
import os
import random
import sys
import time
import numpy as np
import PIL
import torchvision
import torch.utils.data
import pickle
from pathlib import Path

from torch.utils.data import DataLoader
from dataset import datasets
from learning import algorithms
from utils import eval_helper, lin_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluating a trained model on all datasets')
    # training
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--opt_thres_file', type=str,
                        default=Path(os.path.abspath(__file__)).parent/'notebooks'/'opt_thres.pkl')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.multiprocessing.set_sharing_strategy('file_system')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = Path(args.model_dir)
    output_dir = model_path
    old_args = json.load((model_path/'args.json').open('r'))
    loaded = torch.load(model_path/'model.best.pkl')
    hparams = loaded['model_hparams']
    dataset = old_args['dataset']

    print('Training Args:')
    for k, v in sorted(old_args.items()):
        print('\t{}: {}'.format(k, v))

    opt_thress = pickle.load(Path(args.opt_thres_file).open('rb'))
    opt_thres = opt_thress[(dataset[0], hparams['task'], hparams['attr'], old_args['algorithm'])]

    eval_thress = [0.5, opt_thres]
    eval_thress_suffix = ['_50', '_opt']

    def make_combined_dataset(names, sset, group_def, override_attr=None):
        ind_datasets = []
        for ds in names:
            subset_query = 'age not in [0, 4]' if ds in ['ISIC', 'ODIR'] else None
            ind_datasets.append(vars(datasets)[ds](args.data_dir, sset, hparams, group_def=group_def,
                                                   subset_query=subset_query, override_attr=override_attr))
        return datasets.ConcatImageDataset(ind_datasets)

    if len(dataset) == 1:
        if dataset[0] in vars(datasets):
            subset_query = 'age not in [0, 4]' if dataset[0] in ['ISIC', 'ODIR'] else None
            train_dataset = vars(datasets)[dataset[0]](args.data_dir, 'tr', hparams, group_def='group', subset_query=subset_query)
            val_dataset = vars(datasets)[dataset[0]](args.data_dir, 'va', hparams, group_def='group', subset_query=subset_query)
            test_dataset = vars(datasets)[dataset[0]](args.data_dir, 'te', hparams, group_def='group', subset_query=subset_query)
        else:
            raise NotImplementedError
    else:
        train_dataset = make_combined_dataset(dataset, 'tr', 'group')
        val_dataset = make_combined_dataset(dataset, 'va', 'group')
        test_dataset = make_combined_dataset(dataset, 'te', 'group')

    if old_args['algorithm'] == 'DFR':
        train_datasets = []
        for ds in dataset:
            subset_query = 'age not in [0, 4]' if ds in ['ISIC', 'ODIR'] else None
            train_datasets.append(vars(datasets)[ds](
                args.data_dir, 'va', hparams, group_def='group', subsample_type='group', subset_query=subset_query))
        train_dataset = datasets.ConcatImageDataset(train_datasets)

    algorithm_class = algorithms.get_algorithm_class(old_args['algorithm'])
    algorithm = algorithm_class('images', (3, 224, 224), 2, loaded['num_attributes'], 0, hparams,
                                grp_sizes=train_dataset.group_sizes, attr_sizes=train_dataset.attr_sizes).to(device)

    algorithm.load_state_dict(loaded['model_dict'])
    algorithm.eval()

    # evaluate stratified ERM on all samples
    if old_args['algorithm'] == 'StratifiedERM':
        val_dataset = val_dataset.orig_ds
        test_dataset = test_dataset.orig_ds

    num_workers = train_dataset.N_WORKERS

    split_names = ['va', 'te']
    final_eval_loaders = [DataLoader(
        dataset=dset,
        batch_size=max(128, hparams['batch_size'] * 2),
        num_workers=num_workers,
        shuffle=False)
        for dset in [val_dataset, test_dataset]
    ]
    if dataset[0] in datasets.CXR_DATASETS:
        # all CXR datasets with matching tasks
        all_cxr = []
        for ds in datasets.CXR_DATASETS:
            if hparams['task'] in vars(datasets)[ds].TASKS:
                all_cxr.append(ds)
                split_names_ds = vars(datasets)[ds].EVAL_SPLITS
                for attr in vars(datasets)[ds].AVAILABLE_ATTRS:
                    final_eval_loaders += [DataLoader(
                        dataset=dset,
                        batch_size=max(128, hparams['batch_size'] * 2),
                        num_workers=num_workers,
                        shuffle=False)
                        for dset in [vars(datasets)[ds](args.data_dir, split, hparams, override_attr=attr)
                                     for split in split_names_ds]
                    ]
                    for j in split_names_ds:
                        split_names.append(f'{ds}-{attr}-{j}')
        # add compound eval sets
        all_ood = [i for i in all_cxr if i not in dataset]
        print(f"All OOD sets: {str(all_ood)}")
        print(f"All CXR sets: {str(all_cxr)}")
        for combined_set_name, combined_set in zip(['all_ood', 'all_cxr'], [all_ood, all_cxr]):
            for attr in ['sex', 'age']:
                split_names.append(f'{combined_set_name}-{attr}-te')
                eval_ds = make_combined_dataset(combined_set, 'te', 'group', override_attr=attr)
                final_eval_loaders += [DataLoader(
                    dataset=eval_ds,
                    batch_size=max(128, hparams['batch_size'] * 2),
                    num_workers=num_workers,
                    shuffle=False)
                ]
    print("Before eval on all sets", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    final_results = {split: eval_helper.eval_metrics(algorithm, loader, device, add_arrays=True, thress=eval_thress,
                                                     thress_suffix=eval_thress_suffix)
                     for split, loader in zip(split_names, final_eval_loaders)}

    print("Finished eval; Starting representation computation", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    # get representations and train linear predictors
    train_dataset_rep = make_combined_dataset(dataset, 'tr', 'group')  
    train_zs, train_atts, train_ys = lin_eval.get_representations(algorithm, DataLoader(
        dataset=train_dataset_rep,
        batch_size=max(128, hparams['batch_size'] * 2),
        num_workers=num_workers,
        shuffle=False), device)
    val_zs, val_atts, val_ys = lin_eval.get_representations(algorithm, final_eval_loaders[0], device)
    test_zs, test_atts, test_ys = lin_eval.get_representations(algorithm, final_eval_loaders[1], device)

    pickle.dump({
         'va': (val_zs, val_atts, val_ys),
         'te': (test_zs, test_atts, test_ys)
    }, open(os.path.join(output_dir, 'reps.pkl'), 'wb'))

    print("Finished representation computation; Starting LR training", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    lin_eval_metrics = lin_eval.eval_lin_attr_pred(train_zs, train_atts, train_ys,
                                                   val_zs, val_atts, val_ys,
                                                   test_zs, test_atts, test_ys)

    final_results = {**final_results, **lin_eval_metrics, 'opt_thres': opt_thres}
    pickle.dump(final_results, open(os.path.join(output_dir, 'final_results_eval.pkl'), 'wb'))

    with open(os.path.join(output_dir, 'done_eval'), 'w') as f:
        f.write('done')

    print("Finished everything!", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
