import hashlib
import os
import sys
import math
import numpy as np
import operator
import torch
import datetime
import pickle
import json
from numbers import Number
from collections import OrderedDict, Counter


def pickle_save(filename, obj):
    filename = str(filename)
    if sys.platform == 'darwin':
        # Mac Pickle cannot write file larger than 2GB
        return mac_pickle_dump(filename, obj)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f, protocol=4)


def pickle_load(filename):
    filename = str(filename)
    if sys.platform == 'darwin':
        # Mac Pickle cannot read file larger than 2GB
        return mac_pickle_load(filename)
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)


def mac_pickle_load(file_path):
    import os
    max_bytes = 2 ** 31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)


def mac_pickle_dump(filename, obj):
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(obj, protocol=4)
    with open(filename, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def load_json(json_path):
    import json
    assert json_path.exists(), f"{json_path} not found, please create first"
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config


def save_json(json_path, data):
    import json
    with open(json_path, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4, separators=(',', ': '))


class NumpyEncoder(json.JSONEncoder):
    # https://github.com/hmallen/numpyencoder/blob/f8199a61ccde25f829444a9df4b21bcb2d1de8f2/numpyencoder/numpyencoder.py
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, np.bool_):
            return bool(obj)

        elif isinstance(obj, np.void):
            return None

        return json.JSONEncoder.default(self, obj)    


class TextFormat:
    ColorCode = {
        'black':        '\033[30m',
        'darkred':      '\033[31m',
        'darkgreen':    '\033[32m',
        'darkyellow':   '\033[33m',
        'darkblue':     '\033[34m',
        'darkpink':     '\033[35m',
        'darkcyan':     '\033[36m',
        'grey':         '\033[37m',
        'white':        '\033[38m',
        'darkgrey':     '\033[90m',
        'red':          '\033[91m',
        'green':        '\033[92m',
        'yellow':       '\033[93m',
        'blue':         '\033[94m',
        'pink':         '\033[95m',
        'cyan':         '\033[96m',
    }
    StyleCode = {
        'normal':        '\033[0m',
        'bold':          '\033[01m',
        'disable':       '\033[02m',
        'underline':     '\033[04m',
        'reverse':       '\033[07m',
        'strikethrough': '\033[09m',
        'invisible':     '\033[08m',
    }
    EndCode = '\033[0m'

    @classmethod
    def format(cls, text, color='white'):
        return cls.ColorCode[color] + text + cls.EndCode


def log(text, color='white', style='normal', with_time=True, handle=None):
    if with_time:
        text = '[' + datetime.datetime.now().strftime('%H:%M:%S') + '] ' + str(text)
    print(TextFormat.StyleCode[style] + TextFormat.ColorCode[color] + str(text) + TextFormat.EndCode)
    if handle is not None:
        handle.write(str(text) + '\n')
    return text


def print_yellow(text, with_time=True):
    return log(text, color='yellow', with_time=with_time)


def print_cyan(text, with_time=True):
    return log(text, color='cyan', with_time=with_time)


def print_green(text, with_time=True):
    return log(text, color='green', with_time=with_time)


def prepare_folders(args):
    folders_util = [args.output_dir,
                    os.path.join(args.output_dir, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print(f"===> Creating folder: {folder}")
            os.makedirs(folder)


def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()


class MovingAverage:
    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                # correction by 1/(1 - self.ema)
                # so that the gradients amplitude backpropagated in dataset is independent of self.ema
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data


def count_samples_per_class(targets, num_labels):
    counts = Counter()
    for y in targets:
        counts[int(y)] += 1
    return [counts[i] if counts[i] else np.inf for i in range(num_labels)]


def make_balanced_weights_per_sample(targets):
    counts = Counter()
    classes = []
    for y in targets:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)
    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(targets))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


def print_separator():
    print("="*80)


def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.4f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    print(sep.join([format_val(x) for x in row]), end_)


def safe_load(parsed):
    # certain metrics (e.g., AUROC) sometimes saved as a 1-element list
    if isinstance(parsed, list):
        return parsed[0]
    else:
        return parsed


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys

    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of dataset corresponding to a random split of the given dataset,
    with n data points in the first dataset and the rest in the last using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))
        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def mixup_data(x, y, alpha=1., device="cpu"):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def accuracy(network, loader, device):
    num_labels = loader.dataset.num_labels
    num_attributes = loader.dataset.num_attributes
    corrects = torch.zeros(num_attributes * num_labels)
    totals = torch.zeros(num_attributes * num_labels)

    network.eval()
    with torch.no_grad():
        for _, x, y, a in loader:
            p = network.predict(x.to(device))
            p = (p > 0).cpu().eq(y).float() if p.squeeze().ndim == 1 else p.argmax(1).cpu().eq(y).float()
            groups = (num_attributes * y + a)
            for g in groups.unique():
                corrects[g] += p[groups == g].sum()
                totals[g] += (groups == g).sum()
        corrects, totals = corrects.tolist(), totals.tolist()

        total_acc = sum(corrects) / sum(totals)
        group_acc = [c / t if t > 0 else np.inf for c, t in zip(corrects, totals)]
    network.train()

    return total_acc, group_acc


def adjust_learning_rate(optimizer, lr, step, total_steps, schedule, cos=False):
    """Decay the learning rate based on schedule"""
    if cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * step / total_steps))
    else:  # stepwise lr schedule
        for milestone in schedule:
            lr *= 0.1 if step >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


def make_grid(tensor, nrow=8, padding=2, normalize=False, ranges=None, scale_each=False, pad_value=0):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        ranges (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        return tensor
    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if ranges is not None:
            assert isinstance(ranges, tuple), \
                "ranges has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min)

        def norm_range(t, ranges):
            if ranges is not None:
                norm_ip(t, ranges[0], ranges[1])
            else:
                norm_ip(t, t.min(), t.max())

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, ranges)
        else:
            norm_range(tensor, ranges)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2, normalize=False, ranges=None, scale_each=False, pad_value=0):
    from PIL import Image
    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, ranges=ranges, scale_each=scale_each)
    ndarray = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarray)
    im.save(filename)
