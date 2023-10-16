import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler

from util.torch import worker_init_fn


class OrderedDistributedSampler(Sampler):
    """
    Sampler that orders samples in a specified order and assigns them to different processes.

    Attributes:
        dataset (Dataset): The dataset to sample from.
        num_replicas (int): The total number of processes participating in the distributed training.
        rank (int): The rank of the current process.
        num_samples (int): The number of samples per process.
        total_size (int): The total number of samples across all processes.

    Methods:
        __iter__(): Returns an iterator over the indices of the samples for the current process.
        __len__(): Returns the number of samples per process.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        """
        Constructor.

        Args:
            dataset (Dataset): The dataset to sample from.
            num_replicas (int): Number of processes in the distributed training. Defaults to None.
            rank (int): The rank of the current process. Defaults to None.

        Raises:
            AssertionError: If num_replicas or rank is None.
        """
        assert num_replicas is not None
        assert rank is not None
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(np.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        """
        Returns an iterator over the indices of the samples to be used in the current process.
        """
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[
            self.rank * self.num_samples: self.rank * self.num_samples
            + self.num_samples
        ]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        """
        Returns the number of samples per process.
        """
        return self.num_samples


def define_loaders(
    train_dataset,
    val_dataset,
    batch_size=32,
    val_bs=32,
    distributed=False,
    world_size=0,
    local_rank=0,
    num_workers=0,
):
    """
    Define data loaders for training and validation datasets.

    If `distributed` is True, the data loaders will use DistributedSampler for shuffling the
    training dataset and OrderedDistributedSampler for sampling the validation dataset.

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        batch_size (int): The batch size for training data loader. Default to 32.
        val_bs (int): The batch size for validation data loader. Default to 32.
        distributed (bool): Whether to use distributed training. Default to False.
        world_size (int): The total number of processes for distributed training. Default to 0.
        local_rank (int): The rank of the current process. Default to 0.
        num_workers (int): Number of workers to use for the dataloaders. Default to 0.

    Returns:
        tuple: A tuple containing the training data loader and the validation data loader.
    """
    sampler, val_sampler = None, None
    if distributed:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
            seed=world_size + local_rank,
        )

        val_sampler = OrderedDistributedSampler(
            val_dataset, num_replicas=world_size, rank=local_rank
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=None,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=None,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader
