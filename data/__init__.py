from importlib import import_module
from torch.utils.data import DataLoader
import torch
import numpy as np


class Data:
    def __init__(self, config, test_only=False):
        ds_cfg = config["dataset"]
        module_data = import_module("data.rawdata")
        RawData = getattr(module_data, "RawData")
        BatchSampler = getattr(module_data, "PatientBatchSampler")
        if not test_only:
            train_set = RawData(config, mode="train", repeat=ds_cfg["train_repeat"], augment=ds_cfg["augment"])
            val_set = RawData(config, mode="val", repeat=ds_cfg["val_repeat"], augment=False)

            train_batch_sampler = BatchSampler(
                train_set.indices,
                batch_size=ds_cfg["batch_size"],
                repeat=ds_cfg["train_repeat"],
                shuffle=True,
                drop_last=True,
            )
            val_batch_sampler = BatchSampler(
                val_set.indices, batch_size=1, repeat=ds_cfg["val_repeat"], shuffle=False, drop_last=False
            )
            # DataLoader
            self.loader_train = DataLoader(
                train_set,
                batch_sampler=train_batch_sampler,
                num_workers=config["n_threads"],
                pin_memory=True,
            )

            self.loader_val = DataLoader(
                val_set,
                batch_sampler=val_batch_sampler,
                num_workers=config["n_threads"],
                pin_memory=True,
            )

        else:  # test_only
            test_set = RawData(config, mode="test", augment=False)
            self.loader_test = DataLoader(
                test_set,
                batch_size=ds_cfg["batch_size"],
                num_workers=config["n_threads"],
                shuffle=False,
                pin_memory=True,
            )
