#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LabRF & RadChar Data Loader
IEEE TAES (submitted) - "Self-Supervised RF Adaptation for Robust Radar Signal Recognition"
This script loads unlabelled LabRF frames for pre-training, and RadChar-Holdout and the RadChar-nShot draws for fine-tuning and evaluation
Created on May 24, 2026
@author: Zi Huang
"""

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

SUBSETS = ["fm1", "fm2", "sb1", "sb2"]

class LabRFDataset(Dataset):
    def __init__(self, file_path: str):
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as file:
            self.length = file['iq'].shape[0]
            self.attrs = dict(file.attrs)   # Capture settings, e.g. sampling_rate, center_frequency

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Frames are read from disk on demand, as a full subset is up to 4 GB in memory
        with h5py.File(self.file_path, 'r') as file:
            iq = file['iq'][idx]
        frame = {
            'iq_real': torch.tensor(np.real(iq), dtype=torch.float32),   # Real signal part only
            'iq_imag': torch.tensor(np.imag(iq), dtype=torch.float32),   # Imaginary signal part only
            'iq_cplx': torch.tensor(iq, dtype=torch.cfloat),             # Complex signal
        }
        return frame

class RadCharDataset(Dataset):
    def __init__(self, file_path: str, group: str = "/"):
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as file:
            self.data = file[group]['iq'][:]          # Group holding the frames, e.g. "1shot/draw1"
            self.labels = file[group]['labels'][:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame = {
            'iq_real': torch.tensor(np.real(self.data[idx]), dtype=torch.float32),   # Real signal part only
            'iq_imag': torch.tensor(np.imag(self.data[idx]), dtype=torch.float32),   # Imaginary signal part only
            'iq_cplx': torch.tensor(self.data[idx], dtype=torch.cfloat),             # Complex signal
            'label_class': torch.tensor(self.labels[idx][1], dtype=torch.int64),     # Class label
            'snr': torch.tensor(self.labels[idx][-1], dtype=torch.float32),          # SNR value
        }
        return frame

class LabRFDataModule():
    def __init__(self, data_path: str):
        # Path to .h5 data file location
        self.data_path = data_path

    def loadPretrainingSet(self, subset: str):
        """For pre-training without labels"""
        if subset not in SUBSETS:
            raise ValueError(f"Argument subset must be one of {SUBSETS}")
        # NOTE: Download .h5 file from https://www.kaggle.com/datasets/abcxyzi/labrf-taes-2026
        return LabRFDataset(self.data_path + f"LabRF-{subset}.h5")

    def loadFinetuningSet(self, fewshot: int, draw: int):
        """For few-shot fine-tuning on one of ten support-set draws"""
        if fewshot not in [1, 5, 10] or draw not in range(1, 11):
            raise ValueError("Argument fewshot must be one of [1, 5, 10] and draw one of 1 to 10")
        return RadCharDataset(self.data_path + "RadChar-nShot-Draws.h5", group=f"{fewshot}shot/draw{draw}")

    def loadHoldoutSet(self):
        """For evaluation of fine-tuned models"""
        return RadCharDataset(self.data_path + "RadChar-Holdout.h5")

if __name__ == "__main__":
    # NOTE: Change the path as required
    LabRF = LabRFDataModule(data_path="./labrf/")

    # Example usage, to access the I/Q data of the k-th index in each subset
    for subset in SUBSETS:
        ds = LabRF.loadPretrainingSet(subset=subset)
        print("Subset:", subset)
        print("Number of frames:", len(ds))
        print("Sampling rate (Hz):", ds.attrs["sampling_rate"])
        print("Centre frequency (Hz):", ds.attrs["center_frequency"])
        print("Complex I/Q data (shape):", ds[0]["iq_cplx"].shape, "\n")

    # For few-shot fine-tuning and evaluation
    ds_finetune = LabRF.loadFinetuningSet(fewshot=1, draw=1)
    ds_eval = LabRF.loadHoldoutSet()
    for k in range(3):
        print("Frame:", k)
        print("Complex I/Q data (shape):", ds_eval[k]["iq_cplx"].shape)
        print("Corresponding class label:", ds_eval[k]["label_class"])
        print("Corresponding SNR label:", ds_eval[k]["snr"], "\n")
