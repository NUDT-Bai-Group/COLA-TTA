import logging
import math
import os
from PIL import Image
import yaml
import numpy as np

from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms

import augmix_ops as augmentations

from moco.loader import GaussianBlur

import time
import psutil
from contextlib import contextmanager
from collections import defaultdict
from thop import profile, clever_format
import wandb
from typing import Dict, Any, Optional, Tuple
import statistics
import threading
import queue

try:
    from calflops import calculate_flops as calflops_calculate
    CALFLOPS_AVAILABLE = True
except ImportError:
    CALFLOPS_AVAILABLE = False
    calflops_calculate = None

LOG_FORMAT = "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

NUM_CLASSES = {"DomainNet-126": 126, "VISDA-C": 12, "OfficeHome":65,"PACS":7,"VLCS":5,"terra_incognita":10}


def configure_logger(rank, log_path=None):
    if log_path:
        log_dir = os.path.dirname(log_path)
        os.makedirs(log_dir, exist_ok=True)

    # only master process will print & write
    level = logging.INFO if rank in {-1, 0} else logging.WARNING
    handlers = [logging.StreamHandler()]
    if rank in {0, -1} and log_path:
        handlers.append(logging.FileHandler(log_path, "w"))

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=handlers,
        force=True,
    )


class UnevenBatchLoader:
    """Loader that loads data from multiple datasets with different length."""

    def __init__(self, data_loaders, is_ddp=False):
        # register N data loaders with epoch counters.
        self.data_loaders = data_loaders
        self.epoch_counters = [0 for _ in range(len(data_loaders))]

        # set_epoch() needs to be called before creating the iterator
        self.is_ddp = is_ddp
        if is_ddp:
            for data_loader in data_loaders:
                data_loader.sampler.set_epoch(0)
        self.iterators = [iter(data_loader) for data_loader in data_loaders]

    def next_batch(self):
        """Load the next batch by collecting from N data loaders.
        Args:
            None
        Returns:
            data: a list of N items from N data loaders. each item has the format
                output by a single data loader.
        """
        data = []
        for i, iterator in enumerate(self.iterators):
            try:
                batch_i = next(iterator)
            except StopIteration:
                self.epoch_counters[i] += 1
                # create a new iterator
                if self.is_ddp:
                    self.data_loaders[i].sampler.set_epoch(self.epoch_counters[i])
                new_iterator = iter(self.data_loaders[i])
                self.iterators[i] = new_iterator
                batch_i = next(new_iterator)
            data.append(batch_i)

        return data

    def update_loader(self, idx, loader, epoch=None):
        if self.is_ddp and isinstance(epoch, int):
            loader.sampler.set_epoch(epoch)
        self.iterators[idx] = iter(loader)

class CustomDistributedDataParallel(DistributedDataParallel):
    """A wrapper class over DDP that relay "module" attribute."""

    def __init__(self, model, **kwargs):
        super(CustomDistributedDataParallel, self).__init__(model, **kwargs)

    def __getattr__(self, name):
        try:
            return super(CustomDistributedDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def remove_wrap_arounds(tensor, ranks):
    if ranks == 0:
        return tensor

    world_size = dist.get_world_size()
    single_length = len(tensor) // world_size
    output = []
    for rank in range(world_size):
        sub_tensor = tensor[rank * single_length : (rank + 1) * single_length]
        if rank >= ranks:
            output.append(sub_tensor[:-1])
        else:
            output.append(sub_tensor)
    output = torch.cat(output)

    return output


def get_categories(category_file):
    """Return a list of categories ordered by corresponding label.

    Args:
        category_file: str, path to the category file. can be .yaml or .txt

    Returns:
        categories: List[str], a list of categories ordered by label.
    """
    if category_file.endswith(".yaml"):
        with open(category_file, "r") as fd:
            cat_mapping = yaml.load(fd, Loader=yaml.SafeLoader)
        categories = list(cat_mapping.keys())
        categories.sort(key=lambda x: cat_mapping[x])
    elif category_file.endswith(".txt"):
        with open(category_file, "r") as fd:
            categories = fd.readlines()
        categories = [cat.strip() for cat in categories if cat]
    else:
        raise NotImplementedError()

    categories = [cat.replace("_", " ") for cat in categories]
    return categories

    # if dataset_name in ['OfficeHome']:
    #     print('using OfficeHome normalization')
    #     return transforms.Normalize(mean=[0.6302, 0.6068, 0.5821],std=[0.2567, 0.2575, 0.2610])#normalize_OfficeHome

def get_augmentation(aug_type, normalize=None,args=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],std=[0.26862954, 0.26130258, 0.27577711]
        )
    if aug_type == "moco-v2":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "moco-v1":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "plain":
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "clip_inference":
        return transforms.Compose(
            [
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "test0":
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "test":
        return transforms.Compose(
            [
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return None

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def save_checkpoint(model, optimizer, epoch, save_path="checkpoint.pth.tar"):
    state = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, save_path)


def adjust_learning_rate(optimizer, progress, args):
    """
    Decay the learning rate based on epoch or iteration.
    """
    if args.optim.cos:
        decay = 0.5 * (1.0 + math.cos(math.pi * progress / args.learn.full_progress))
    elif args.optim.exp:
        decay = (1 + 10 * progress / args.learn.full_progress) ** -0.75
    else:
        decay = 1.0
        for milestone in args.optim.schedule:
            decay *= args.optim.gamma if progress >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay

    return decay


def per_class_accuracy(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    acc_per_class = (matrix.diagonal() / matrix.sum(axis=1) * 100.0).round(2)
    logging.info(
        f"Accuracy per class: {acc_per_class}, mean: {acc_per_class.mean().round(2)}"
    )

    return acc_per_class


def get_distances(X, Y, dist_type="euclidean"):
    """
    Args:
        X: (N, D) tensor
        Y: (M, D) tensor
    """
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances


def is_master(args):
    return args.rank % args.ngpus_per_node == 0


def use_wandb(args):
    return is_master(args) and args.use_wandb

def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
    ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)

    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug) 
    mix = m * x_processed + (1 - m) * mix
    return [x_processed, mix]


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        all_views = ([image] + views)
        return torch.stack(all_views)


def get_Augmixtransforms(normalize=None):
    if not normalize:
        # normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # )
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC), # type: ignore
            transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    transform = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=False)
    return transform

    try:
        if not torch.cuda.is_available():
            if stage_name:
                logging.info(f"[{stage_name}] CUDA not available")
            return
        
        device = torch.cuda.current_device()
        
        # 基础内存信息
        allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(device) / (1024**3)   # GB
        allocated_mb = torch.cuda.memory_allocated(device) / (1024**2)  # GB
        reserved_mb = torch.cuda.memory_reserved(device) / (1024**2)   # GB
        
        # 获取激活显存信息
        try:
            memory_stats = torch.cuda.memory_stats(device)
            active_bytes = memory_stats.get('active_bytes.all.current', 0)
            active_gb = active_bytes / (1024**3)
            active_mb = active_bytes / (1024**2)
            
            if stage_name:
                logging.info(f"[{stage_name}] GPU Memory - "
                           f"Allocated: {allocated:.2f}GB ({allocated_mb:.0f}MB), "
                           f"Reserved: {reserved:.2f}GB ({reserved_mb:.0f}MB), "
                           f"Active: {active_gb:.2f}GB ({active_mb:.0f}MB)")
            
        except Exception:
            if stage_name:
                logging.info(f"[{stage_name}] GPU Memory - "
                           f"Allocated: {allocated:.2f}GB ({allocated_mb:.0f}MB), "
                           f"Reserved: {reserved:.2f}GB ({reserved_mb:.0f}MB)")
        
    except Exception as e:
        logging.debug(f"GPU memory check failed: {e}")