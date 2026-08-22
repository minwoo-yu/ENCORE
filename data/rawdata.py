import os
from data import common
import numpy as np
import torch.utils.data as data
import math
import torch
import random
from collections import defaultdict


class PatientBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, patient_slice, batch_size, repeat, shuffle=True, drop_last=True):
        self.patient_slice = patient_slice
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.repeat = repeat
        patient_slice_lengths = [math.ceil(len(indices) / batch_size) for indices in patient_slice.values()]
        self.iterations = int(sum(patient_slice_lengths) * repeat)
        self.patient_ids = list(patient_slice.keys())

    def __iter__(self):
        if self.shuffle:
            for _ in range(self.iterations):
                pid = random.choice(self.patient_ids)
                indices = self.patient_slice[pid]
                batch = random.sample(indices, self.batch_size)
                yield batch
        else:
            for pid in self.patient_ids * self.repeat:
                indices = self.patient_slice[pid]
                for i in range(0, len(indices), self.batch_size):
                    batch = indices[i : i + self.batch_size]
                    if len(batch) < self.batch_size and self.drop_last:
                        continue
                    yield batch

    def __len__(self):
        return self.iterations


class RawData(data.Dataset):
    def __init__(self, config, mode="train", repeat=1, augment=False):
        self.dataset_spec = config["dataset"]
        self.mode = mode
        self.augment = augment
        if mode == "train":
            self.noisy, self.metadata, self.indices = self._scan()
            patient_slice_lengths = [
                math.ceil(len(indices) / self.dataset_spec["batch_size"]) for indices in self.indices.values()
            ]
            self.iterations = int(sum(patient_slice_lengths) * repeat)
        elif mode == "val":
            self.clean, self.metadata, self.indices = self._scan()
            patient_slice_lengths = [
                math.ceil(len(indices) / self.dataset_spec["batch_size"]) for indices in self.indices.values()
            ]
            self.iterations = int(sum(patient_slice_lengths) * repeat)
        elif mode == "test":
            self.noisy, self.clean, self.metadata = self._scan()

    def __getitem__(self, idx):
        if self.mode == "train":
            noisy = self.noisy[idx].astype(np.float32)
            metadata = self.metadata[idx]
            noisy, pos = self.preparation(noisy)
            return np.expand_dims(noisy, 0), pos.astype(np.float32), metadata
        elif self.mode == "val":
            clean = self.clean[idx].astype(np.float32)
            metadata = self.metadata[idx]
            return np.expand_dims(clean, 0), metadata
        else:
            noisy = self.noisy[idx].astype(np.float32)
            clean = self.clean[idx].astype(np.float32)
            metadata = self.metadata[idx].astype(np.float32)
            return np.expand_dims(noisy, 0), np.expand_dims(clean, 0), metadata

    def __len__(self):
        return len(self.clean)

    def _scan(self):
        if self.mode in ("train", "val"):
            proj_data = []
            metadata_list = []
            indices_map = defaultdict(list)
            base = self.dataset_spec["train_data_path"]
            patients = self.dataset_spec[f"{self.mode}_patient"]
            current_idx = 0

            for pid in patients:
                if self.mode == "train" and self.dataset_spec["mode"] != "N2C":
                    folder = os.path.join(base, pid, "noisy") 
                else:
                    folder = os.path.join(base, pid, "clean")
                files = sorted(os.listdir(folder))
                for file in files:
                    loadfile = np.load(os.path.join(folder, file), allow_pickle=True).item()
                    proj_data.append(loadfile["projection_data"])
                    metadata_list.append(loadfile["metadata"])
                    indices_map[pid].append(current_idx)
                    current_idx += 1

            projections = np.stack(proj_data, axis=0)
            return projections, metadata_list, indices_map
        else:
            noisy_data, clean_data, metadata = [], [], []
            base = self.dataset_spec["test_data_path"]
            patients = self.dataset_spec["test_patient"]
            for pid in patients:
                folder_noisy = os.path.join(base, pid, "noisy", str(self.dataset_spec["dose_level"]))
                folder_clean = os.path.join(base, pid, "clean")
                files = sorted(os.listdir(folder_noisy))
                for file in files:
                    noisy_data.append(
                        np.load(os.path.join(folder_noisy, file), allow_pickle=True).item()["projection_data"]
                    )
                    loadfile = np.load(os.path.join(folder_clean, file), allow_pickle=True).item()
                    clean_data.append(loadfile["projection_data"])
                    metadata.append(loadfile["metadata"])
            noisy = np.stack(noisy_data, axis=0)
            clean = np.stack(clean_data, axis=0)
            metadata = np.stack(metadata, axis=0)
            return noisy, clean, metadata

    def preparation(self, clean):
        if self.augment:
            clean = common.augment(clean)
        pos = common.patch_position(self.dataset_spec["patch_size"])
        return clean, pos
