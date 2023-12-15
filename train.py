import argparse
import collections
import json
import os
import random
import sys
import time
import numpy as np
import pandas as pd
import PIL
import torch
import torchvision
import torch.utils.data
from tensorboard_logger import Logger
import pickle
from pathlib import Path

from torch.utils.data import DataLoader
from dataset import datasets
import hparams_registry
from learning import algorithms, early_stopping, swad_utils
from utils import misc, eval_helper
from dataset.fast_dataloader import InfiniteDataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shortcut Learning in Chest X-rays')
    # training
    parser.add_argument('--store_name', type=str, default='debug')
    parser.add_argument('--dataset', type=str, default=["MIMIC"], nargs='+')
    parser.add_argument('--task', type=str, default="No Finding", choices=datasets.TASKS + datasets.ATTRS)
    parser.add_argument('--attr', type=str, default="sex", choices=datasets.ATTRS)
    parser.add_argument('--group_def', type=str, default="group", choices=['group', 'label'])
    parser.add_argument('--algorithm', type=str, default="ERM", choices=algorithms.ALGORITHMS)
    # others
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for random hparams (0 for "default hparams")')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None)    
    parser.add_argument('--log_online', help='Log online using wandb', action='store_true')
    parser.add_argument('--skip_ood_eval', help='skip evals on OOD datasets', action='store_true')
    parser.add_argument('--log_all', help='Log all val metrics at each step to tb and wandb', action='store_true')
    parser.add_argument('--stratified_erm_subset', type=int, default=None)
    # two-stage related
    parser.add_argument('--stage1_folder', type=str)
    # early stopping
    parser.add_argument('--use_es', action='store_true')
    parser.add_argument('--es_strategy', choices=['metric'], default='metric')
    parser.add_argument('--es_metric', type=str, default='min_group:accuracy')
    parser.add_argument('--es_patience', type=int, default=5, help='Stop after this many checkpoints w/ no improvement')
    # checkpoints
    parser.add_argument('--resume', '-r', type=str, default='')
    parser.add_argument('--checkpoint_freq', type=int, default=None, help='Checkpoint every N steps')
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--debug', action='store_true')
    # architectures and pre-training sources
    parser.add_argument('--image_arch', default='densenet_sup_in1k',
                        choices=['densenet_sup_in1k', 'resnet_sup_in1k', 'resnet_sup_in21k', 'resnet_simclr_in1k',
                                 'resnet_barlow_in1k', 'vit_sup_in1k', 'vit_sup_in21k', 'vit_sup_swag', 'vit_clip_oai',
                                 'vit_clip_laion', 'vit_dino_in1k', 'resnet_dino_in1k'])
    # data augmentations
    parser.add_argument('--aug', default='basic2',
                        choices=['none', 'basic', 'basic2', 'auto_aug', 'rand_aug', 'trivial_aug', 'augmix'])
    args = parser.parse_args()

    start_step = 0
    misc.prepare_folders(args)
    output_dir = os.path.join(args.output_dir, args.store_name)
    if not args.debug:
        sys.stdout = misc.Tee(os.path.join(output_dir, 'out.txt'))
        sys.stderr = misc.Tee(os.path.join(output_dir, 'err.txt'))

    tb_logger = Logger(logdir=output_dir, flush_secs=2)

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

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, misc.seed_hash(args.hparams_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    hparams.update({
        'image_arch': args.image_arch,
        'data_augmentation': args.aug,
        'task': args.task,
        'attr': args.attr,
        'group_def': args.group_def
    })

    if args.log_online:
        import wandb
        import hashlib
        wandb.init(project='subpop_fairness', config={**vars(args), **hparams},
                   name=f"train_{args.dataset}_{args.task}_{args.algorithm}_{args.attr}_"
                        f"{hashlib.md5(str({**vars(args), **hparams}).encode('utf-8')).hexdigest()[:8]}_"
                        f"{os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ else ''}")

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.multiprocessing.set_sharing_strategy('file_system')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def make_combined_dataset(names, sset, group_def, override_attr=None):
        ind_datasets = []
        for ds in names:
            ind_datasets.append(vars(datasets)[ds](args.data_dir, sset, hparams, group_def=group_def, override_attr=override_attr))
        return datasets.ConcatImageDataset(ind_datasets)

    if len(args.dataset) == 1:
        if args.dataset[0] in vars(datasets):
            train_dataset = vars(datasets)[args.dataset[0]](args.data_dir, 'tr', hparams, group_def=args.group_def)
            val_dataset = vars(datasets)[args.dataset[0]](args.data_dir, 'va', hparams, group_def='group')
            test_dataset = vars(datasets)[args.dataset[0]](args.data_dir, 'te', hparams, group_def='group')
        else:
            raise NotImplementedError
    else:
        train_dataset = make_combined_dataset(args.dataset, 'tr', args.group_def)
        val_dataset = make_combined_dataset(args.dataset, 'va', 'group')
        test_dataset = make_combined_dataset(args.dataset, 'te', 'group')

    if args.algorithm == 'DFR':
        train_datasets = []
        for ds in args.dataset:
            train_datasets.append(vars(datasets)[ds](
                args.data_dir, 'va', hparams, group_def=args.group_def, subsample_type='group'))
        train_dataset = datasets.ConcatImageDataset(train_datasets)
    elif args.algorithm == 'StratifiedERM':
        assert args.stratified_erm_subset is not None
        train_dataset = datasets.SubsetImageDataset(
            train_dataset, idxs=np.argwhere(np.array(train_dataset.a) == args.stratified_erm_subset).squeeze())
        val_dataset = datasets.SubsetImageDataset(
            val_dataset, idxs=np.argwhere(np.array(val_dataset.a) == args.stratified_erm_subset).squeeze())
        test_dataset = datasets.SubsetImageDataset(
            test_dataset, idxs=np.argwhere(np.array(test_dataset.a) == args.stratified_erm_subset).squeeze())

    num_workers = train_dataset.N_WORKERS
    input_shape = train_dataset.INPUT_SHAPE
    num_labels = train_dataset.num_labels
    num_attributes = train_dataset.num_attributes
    data_type = train_dataset.data_type
    n_steps = args.steps or train_dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or train_dataset.CHECKPOINT_FREQ

    hparams.update({
        "steps": n_steps
    })
    print(f"Dataset:\n\t[train]\t{len(train_dataset)}"
          f"\n\t[val]\t{len(val_dataset)}")

    if hparams['group_balanced']:
        # if attribute not available, groups degenerate to classes
        train_weights = np.asarray(train_dataset.weights_g)
        train_weights /= np.sum(train_weights)
    elif hparams['attr_balanced']:
        train_weights = np.asarray(train_dataset.weights_a)
        train_weights /= np.sum(train_weights)
    else:
        train_weights = None

    train_loader = InfiniteDataLoader(
        dataset=train_dataset,
        weights=train_weights,
        batch_size=min(len(train_dataset), hparams['batch_size']),
        num_workers=num_workers
    )
    split_names = ['va', 'te']
    eval_loaders = [DataLoader(
        dataset=dset,
        batch_size=max(128, hparams['batch_size'] * 2),
        num_workers=num_workers,
        shuffle=False)
        for dset in [val_dataset, test_dataset]
    ]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(data_type, input_shape, num_labels, num_attributes, len(train_dataset), hparams,
                                grp_sizes=train_dataset.group_sizes, attr_sizes=train_dataset.attr_sizes)

    es_group = args.es_metric.split(':')[0]
    es_metric = args.es_metric.split(':')[1]
    es = early_stopping.EarlyStopping(
        patience=args.es_patience, lower_is_better=early_stopping.lower_is_better[es_metric])
    best_model_path = os.path.join(output_dir, 'model.best.pkl')

    # load stage1 model if using 2-stage algorithm
    if 'CRT' in args.algorithm or 'DFR' in args.algorithm:
        assert os.path.isdir(args.stage1_folder)
        if (Path(args.stage1_folder)/'model.best.pkl').is_file():
            weight_location = Path(args.stage1_folder)/'model.best.pkl'
        else:
            weight_location = Path(args.stage1_folder)/'model.pkl'

        checkpoint = torch.load(weight_location, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_dict'].items():
            if 'classifier' not in k and 'network.1.' not in k:
                new_state_dict[k] = v
        algorithm.load_state_dict(new_state_dict, strict=False)
        print(f"===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]")
        print(f"===> Pre-trained model loaded: '{weight_location}'")

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"===> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_step = checkpoint['start_step']
            args.best_val_acc = checkpoint['best_val_acc']
            algorithm.load_state_dict(checkpoint['model_dict'])
            es = checkpoint['early_stopper']
            print(f"===> Loaded checkpoint '{args.resume}' (step [{start_step}])")
        else:
            print(f"===> No checkpoint found at '{args.resume}'")

    algorithm.to(device)

    train_minibatches_iterator = iter(train_loader)
    checkpoint_vals = collections.defaultdict(lambda: [])
    steps_per_epoch = len(train_dataset) / hparams['batch_size']

    def save_checkpoint(save_dict, filename='model.pkl'):
        if args.skip_model_save:
            return
        filename = os.path.join(output_dir, filename)
        torch.save(save_dict, filename)

    def log_value(key, val, s):
        tb_logger.log_value(key, val, s)
        if args.log_online:
            wandb.log({key: val}, step=s)

    last_results_keys = None
    for step in range(start_step, n_steps):
        if args.use_es and es.early_stop:
            print(f"Early stopping at step {step} with best {args.es_metric}={es.best_score}.")
            break
        step_start_time = time.time()
        i, x, y, a = next(train_minibatches_iterator)
        minibatch_device = (i, x.to(device), y.to(device), a.to(device))

        algorithm.train()
        step_vals = algorithm.update(minibatch_device, step)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if ((not args.debug) and step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }
            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            curr_metrics = {split: eval_helper.eval_metrics(algorithm, loader, device)
                            for split, loader in zip(split_names, eval_loaders)}
            full_val_metrics = curr_metrics['va']

            for split in sorted(split_names):
                results[f'{split}_avg_acc'] = curr_metrics[split]['overall']['accuracy_50']
                results[f'{split}_overall_auroc'] = curr_metrics[split]['overall']['AUROC']
                results[f'{split}_worst_acc'] = curr_metrics[split]['min_group']['accuracy_50']

            results_keys = list(results.keys())
            if results_keys != last_results_keys:
                print("\n")
                misc.print_row([key for key in results_keys if key not in {'mem_gb', 'step_time'}], colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys if key not in {'mem_gb', 'step_time'}], colwidth=12)

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results.update({
                'hparams': hparams,
                'args': vars(args),
            })
            results.update(curr_metrics)

            epochs_path = os.path.join(output_dir, 'results.json')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True, cls=misc.NumpyEncoder) + "\n")

            save_dict = {
                "args": vars(args),
                "best_es_metric": es.best_score,
                "start_step": step + 1,
                "num_labels": num_labels,
                "num_attributes": train_dataset.num_attributes,
                "model_input_shape": input_shape,
                "model_hparams": hparams,
                "model_dict": algorithm.state_dict(),
                "early_stopper": es,
            }
            save_checkpoint(save_dict)

            # logging
            for key in checkpoint_vals.keys() - {'step_time'}:
                log_value(key, results[key], step)
            for key in split_names:
                log_value(f"{key}/avg_acc", results[f"{key}_avg_acc"], step)
                log_value(f"{key}/worst_acc", results[f"{key}_worst_acc"], step)
            if args.log_all:
                for key1 in full_val_metrics:
                    for key2 in full_val_metrics[key1]:
                        if isinstance(full_val_metrics[key1][key2], dict):
                            for key3 in full_val_metrics[key1][key2]:
                                log_value(f"va/{key1}_{key2}_{key3}", full_val_metrics[key1][key2][key3], step)
                        else:
                            log_value(f"va/{key1}_{key2}", full_val_metrics[key1][key2], step)
            if hasattr(algorithm, 'optimizer'):
                log_value('learning_rate', algorithm.optimizer.param_groups[0]['lr'], step)

            if args.use_es:
                if args.es_strategy == 'metric':
                    es_metric_val = full_val_metrics[es_group][es_metric]

                es(es_metric_val, step, save_dict, best_model_path)
                log_value('es_metric', es_metric_val, step)

            checkpoint_vals = collections.defaultdict(lambda: [])

    # load best model and get metrics on eval sets
    if args.use_es and not args.skip_model_save:
        algorithm.load_state_dict(torch.load(os.path.join(output_dir, "model.best.pkl"))['model_dict'])

    algorithm.eval()

    # evaluate stratified ERM on all samples
    if args.algorithm == 'StratifiedERM':
        val_dataset = val_dataset.orig_ds
        test_dataset = test_dataset.orig_ds

    split_names = ['va', 'te']
    final_eval_loaders = [DataLoader(
        dataset=dset,
        batch_size=max(128, hparams['batch_size'] * 2),
        num_workers=num_workers,
        shuffle=False)
        for dset in [val_dataset, test_dataset]
    ]
    if (not args.skip_ood_eval) and (args.dataset[0] in datasets.CXR_DATASETS):
        # all CXR datasets with matching tasks
        all_cxr = []
        for ds in datasets.CXR_DATASETS:
            if args.task in vars(datasets)[ds].TASKS:
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
        all_ood = [i for i in all_cxr if i not in args.dataset]
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
    final_results = {split: eval_helper.eval_metrics(algorithm, loader, device, add_arrays=True)
                     for split, loader in zip(split_names, final_eval_loaders)}
    if args.use_es:
        final_results['es_step'] = es.step

    pickle.dump(final_results, open(os.path.join(output_dir, 'final_results.pkl'), 'wb'))
    print("\nTest accuracy (best validation checkpoint):")
    print(f"\tmean:\t[{final_results['te']['overall']['accuracy_50']:.3f}]\n"
          f"\tworst:\t[{final_results['te']['min_group']['accuracy_50']:.3f}]")
    if args.log_online:
        for split in final_results:
            for key1 in final_results[split]:
                if isinstance(final_results[split][key1], dict):
                    for key2 in final_results[split][key1]:
                        if isinstance(final_results[split][key1][key2], dict):
                            pass
                        else:
                            wandb.log({f"{split}/best_{key1}_{key2}": final_results[split][key1][key2]})

    print("Group-wise accuracy:")
    for split in final_results.keys():
        if split != 'es_step' and not split.startswith('lin_'):
            print('\t[{}] group-wise {}'.format(
                split, (np.array2string(
                    pd.DataFrame(final_results[split]['per_group']).T['accuracy_50'].values,
                    separator=', ', formatter={'float_kind': lambda x: "%.3f" % x}))))

    with open(os.path.join(output_dir, 'done'), 'w') as f:
        f.write('done')
