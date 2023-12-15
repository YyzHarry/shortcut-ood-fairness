import numpy as np
from utils import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    IMAGE_DATASETS = ['MIMIC', 'CheXpert', 'NIH', 'PadChest']

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert name not in hparams
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions

    _hparam('resnet18', False, lambda r: False)
    # nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False, lambda r: bool(r.choice([False, False])))

    if algorithm in ['ReSample', 'CRT']:
        _hparam('group_balanced', True, lambda r: True)
    else:
        _hparam('group_balanced', False, lambda r: False)

    if algorithm in ['ReSampleAttr']:
        _hparam('attr_balanced', True, lambda r: True)
    else:
        _hparam('attr_balanced', False, lambda r: False)

    # Algorithm-specific hparam definitions
    # Each block of code below corresponds to one algorithm

    if algorithm == 'CBLoss':
        _hparam('beta', 0.9999, lambda r: 1 - 10**r.uniform(-5, -2))

    elif algorithm == 'Focal':
        _hparam('gamma', 1, lambda r: 0.5 * 10**r.uniform(0, 1))

    elif algorithm == 'LDAM':
        _hparam('max_m', 0.5, lambda r: 10**r.uniform(-1, -0.1))
        _hparam('scale', 30., lambda r: r.choice([10., 30.]))

    elif algorithm == "IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500, lambda r: int(10**r.uniform(0, 4)))

    elif "Mixup" in algorithm:
        _hparam('mixup_alpha', 0.2, lambda r: 10**r.uniform(-1, 1))

    elif "GroupDRO" in algorithm:
        _hparam('groupdro_eta', 1e-2, lambda r: 10**r.uniform(-3, -1))

    elif algorithm in ["MMD", "CORAL"]:
        _hparam('mmd_gamma', 1., lambda r: 10**r.uniform(-1, 1))

    elif 'DANN' in algorithm:
        _hparam('lambda', 1.0, lambda r: 10**r.uniform(-2, 2))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('weight_decay_g', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('d_steps_per_g_step', 1, lambda r: int(2**r.uniform(0, 3)))
        _hparam('grad_penalty', 0., lambda r: 10**r.uniform(-2, 1))
        _hparam('mlp_width', 256, lambda r: int(2**r.uniform(7, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))

    elif algorithm == 'CVaRDRO':
        _hparam('joint_dro_alpha', 0.1, lambda r: 10**r.uniform(-2, 0))

    elif algorithm == 'JTT':
        _hparam('first_stage_step_frac', 0.5, lambda r: r.uniform(0.2, 0.8))
        _hparam('jtt_lambda', 10, lambda r: 10**r.uniform(0, 2.5))

    elif algorithm == 'LISA':
        _hparam('LISA_alpha', 2., lambda r: 10**r.uniform(-1, 1))
        _hparam('LISA_p_sel', 0.5, lambda r: r.uniform(0, 1))
        _hparam('LISA_mixup_method', 'mixup', lambda r: r.choice(['mixup', 'cutmix']))

    elif algorithm == 'DFR':
        _hparam('dfr_reg', .1, lambda r: 10**r.uniform(-2, 0.5))

    elif algorithm == 'MA':
        _hparam('ma_start_iter', 1000, lambda r: int(1000 * r.choice(range(1, 6))))

    elif algorithm == 'SAM':
        _hparam('sam_rho', 0.05, lambda r: r.choice([0.01, 0.02, 0.05, 0.1]))

    elif algorithm == 'SWA':
        _hparam('swa_start', 500, lambda r: int(100 * r.uniform(5, 10)))
        _hparam('swa_lr', 5e-5, lambda r: 10**r.uniform(-4.5, -4))
        _hparam('swa_anneal_steps', 500, lambda r: int(100 * r.uniform(5, 10)))
        _hparam('swa_update_steps', 500, lambda r: int(100 * r.uniform(5, 10)))

    elif algorithm == 'SWAD':
        _hparam('swad_n_converge', 3, lambda r: r.randint(2, 8))
        _hparam('swad_n_tolerance', 6, lambda r: r.randint(4, 16))
        _hparam('swad_tolerance_ratio', 0.3, lambda r: r.uniform(0.2, 0.4))

    # Dataset-and-algorithm-specific hparam definitions
    # Each block of code below corresponds to exactly one hparam. Avoid nested conditionals

    _hparam('pretrained', True, lambda r: True)
    _hparam('optimizer', 'adam', lambda r: 'adam')
    _hparam('last_layer_dropout', 0., lambda r: 0.)
    _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4, -2))
    _hparam('weight_decay', 1e-4, lambda r: 10**r.uniform(-6, -3))
    _hparam('batch_size', 64, lambda r: 64)

    if 'DANN' in algorithm:
        _hparam('lr_g', 1e-3, lambda r: 10**r.uniform(-4, -2))
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4, -2))

    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
