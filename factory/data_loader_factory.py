import torch

from torch.utils.data import DataLoader

from sl_prioritized_sampling.factory.base_factory import BaseFactory
from sl_prioritized_sampling.dataset.custom_sampler import CustomWeightedRandomSampler


class DataLoaderFactory(BaseFactory):
    @classmethod
    def create(cls, **kwargs):
        train_dataset, val_dataset = kwargs["train_dataset"], kwargs["val_dataset"]

        if bool(kwargs["config"].getint("data_loader", "prioritized_sampler")):
            c_const = kwargs["config"].getfloat("data_loader", "c_constant")
            explore_type = kwargs["config"].get("data_loader", "explore_type")
            exploit_type = kwargs["config"].get("data_loader", "exploit_type")
            sampler = CustomWeightedRandomSampler(c_const=c_const,
                                                  explore_type=explore_type,
                                                  exploit_type=exploit_type,
                                                  num_samples=len(train_dataset.data))
        else:
            sampler = None

        cuda_available = torch.cuda.is_available()
        device = 'cuda' if cuda_available else ''

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=kwargs["config"].getint("data_loader", "batch_size_train"),
                                  num_workers=kwargs["config"].getint("data_loader", "num_workers"),
                                  sampler=sampler,
                                  pin_memory=cuda_available,
                                  pin_memory_device=device,
                                  )

        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=kwargs["config"].getint("data_loader", "batch_size_train"),
                                num_workers=kwargs["config"].getint("data_loader", "num_workers"),
                                pin_memory=cuda_available,
                                pin_memory_device=device
                                )

        return train_loader, val_loader, sampler
