import os
import re
import copy
import time
import torch
import random
import datetime
import numpy as np


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results.

    Args:
        seed (int): Number of the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False  # True
    torch.backends.cudnn.benchmark = True  # False


def save_model_weights(model, filename, verbose=1, cp_folder=""):
    """
    Saves the weights of a PyTorch model.

    Args:
        model (torch model): Model to save the weights of.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        cp_folder (str, optional): Folder to save to. Defaults to "".
    """
    if verbose:
        print(f"\n -> Saving weights to {os.path.join(cp_folder, filename)}\n")
    torch.save(model.state_dict(), os.path.join(cp_folder, filename))


def load_model_weights(model, filename, verbose=1, cp_folder="", strict=True):
    """
    Loads the weights of a PyTorch model. The exception handles cpu/gpu incompatibilities.

    Args:
        model (torch model): Model to load the weights to.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        cp_folder (str, optional): Folder to load from. Defaults to "".
        strict (str, optional): Whether to use strict weight loading. Defaults to True.

    Returns:
        torch model: Model with loaded weights.
    """
    state_dict = torch.load(os.path.join(cp_folder, filename), map_location="cpu")

    try:
        try:
            model.load_state_dict(state_dict, strict=strict)
        except BaseException:
            state_dict_ = {}
            for k, v in state_dict.items():
                state_dict_[re.sub("module.", "", k)] = v
            model.load_state_dict(state_dict_, strict=strict)

    except BaseException:
        try:  # REMOVE CLASSIFIER
            state_dict_ = copy.deepcopy(state_dict)
            try:
                del (
                    state_dict_["encoder.classifier.weight"],
                    state_dict_["encoder.classifier.bias"],
                )
            except KeyError:
                del (
                    state_dict_["encoder.head.fc.weight"],
                    state_dict_["encoder.head.fc.bias"],
                )
            model.load_state_dict(state_dict_, strict=strict)
        except BaseException:  # REMOVE LOGITS
            try:
                for k in ["logits.weight", "logits.bias"]:
                    try:
                        del state_dict[k]
                    except KeyError:
                        pass
                model.load_state_dict(state_dict, strict=strict)
            except BaseException:
                del state_dict["encoder.conv_stem.weight"]
                model.load_state_dict(state_dict, strict=strict)

    if verbose:
        print(
            f"\n -> Loading encoder weights from {os.path.join(cp_folder,filename)}\n"
        )

    return model


def count_parameters(model, all=False):
    """
    Count the parameters of a model.

    Args:
        model (torch model): Model to count the parameters of.
        all (bool, optional):  Whether to count not trainable parameters. Defaults to False.

    Returns:
        int: Number of parameters.
    """

    if all:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def worker_init_fn(worker_id):
    """
    Handles PyTorch x Numpy seeding issues.

    Args:
        worker_id (int]): Id of the worker.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def sync_across_gpus(t, world_size):
    """
    Synchronizes predictions accross all gpus.

    Args:
        t (torch tensor): Tensor to synchronzie
        world_size (int): World size.

    Returns:
        torch tensor: Synced tensor.
    """
    torch.distributed.barrier()
    gather_t_tensor = [torch.ones_like(t) for _ in range(world_size)]
    torch.distributed.all_gather(gather_t_tensor, t)
    return torch.cat(gather_t_tensor)


def init_distributed(cfg):
    """
    Initializes stuff for torch distributed training.

    Args:
        cfg (Config): Config.
    """
    cfg.distributed = False
    if "WORLD_SIZE" in os.environ:
        cfg.distributed = int(os.environ["WORLD_SIZE"]) > 1

    if cfg.distributed:
        cfg.local_rank = int(os.environ["LOCAL_RANK"])

        if cfg.local_rank == 0:
            print("- Training in distributed mode with multiple GPUs.")
        time.sleep(1)

        device = "cuda:%d" % cfg.local_rank
        cfg.device = device

        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=180),
        )
        cfg.world_size = torch.distributed.get_world_size()
        cfg.rank = torch.distributed.get_rank()

        print(
            f"Process {cfg.rank}/{cfg.world_size} - device {device} -  local rank {cfg.local_rank}"
        )

        # syncing the random seed
        cfg.seed = int(
            sync_across_gpus(torch.Tensor([cfg.seed]).to(device), cfg.world_size)
            .detach()
            .cpu()
            .numpy()[0]
        )

    else:
        print("- Training with one GPU.")
        cfg.local_rank = 0
        cfg.world_size = 1
        cfg.rank = 0  # global rank
        device = "cuda:0"
        cfg.device = device
