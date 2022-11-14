import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data


import numpy as np


class Trajectory(data.Dataset):
    def __init__(self, normal_data, emg_data, max_conv_size=7):
        # max_conv_size: always set to a odd number

        self.conv_size = max_conv_size

        self.normal_speed = np.array(normal_data["Speed"], dtype=float)
        self.normal_sin = np.sin(np.array(normal_data["Bearing"], dtype=float))
        self.normal_cos = np.cos(np.array(normal_data["Bearing"], dtype=float))

        self.emg_speed = np.array(emg_data["Speed"], dtype=float)
        self.emg_sin = np.sin(np.array(emg_data["Bearing"], dtype=float))
        self.emg_cos = np.cos(np.array(emg_data["Bearing"], dtype=float))

        self.normal_len = len(self.normal_speed) - (self.conv_size // 2) - 1
        self.emg_len = len(self.emg_speed) - (self.conv_size // 2) - 1

        zeros = np.zeros((self.conv_size // 2))

        self.normal_speed = np.concatenate((zeros, self.normal_speed), axis=0)
        self.normal_sin = np.concatenate((zeros, self.normal_sin), axis=0)
        self.normal_cos = np.concatenate((zeros, self.normal_cos), axis=0)

        self.emg_speed = np.concatenate((zeros, self.emg_speed), axis=0)
        self.emg_sin = np.concatenate((zeros, self.emg_sin), axis=0)
        self.emg_cos = np.concatenate((zeros, self.emg_cos), axis=0)

    def __getitem__(self, index):
        save_index = index

        # pick sequence
        sample_seq_speed = self.normal_speed
        sample_seq_sin = self.normal_sin
        sample_seq_cos = self.normal_cos

        if index >= self.normal_len:
            index -= self.normal_len
            sample_seq_speed = self.emg_speed
            sample_seq_sin = self.emg_sin
            sample_seq_cos = self.emg_cos

        index += self.conv_size // 2

        sample = {}

        sample["index"] = index
        sample["save_index"] = save_index

        seq_speed = sample_seq_speed[index - self.conv_size // 2 : index + self.conv_size // 2 + 1]
        seq_sin = sample_seq_sin[index - self.conv_size // 2 : index + self.conv_size // 2 + 1]
        seq_cos = sample_seq_cos[index - self.conv_size // 2 : index + self.conv_size // 2 + 1]

        label_speed = sample_seq_speed[index + self.conv_size // 2 + 1]
        label_sin = sample_seq_sin[index + self.conv_size // 2 + 1]
        label_cos = sample_seq_cos[index + self.conv_size // 2 + 1]

        sample["train_speed"] = torch.tensor(seq_speed, dtype=torch.float32).unsqueeze(0)
        sample["train_sin"] = torch.tensor(seq_sin, dtype=torch.float32).unsqueeze(0)
        sample["train_cos"] = torch.tensor(seq_cos, dtype=torch.float32).unsqueeze(0)

        sample["label_speed"] = torch.tensor(label_speed, dtype=torch.float32).unsqueeze(0)
        sample["label_sin"] = torch.tensor(label_sin, dtype=torch.float32).unsqueeze(0)
        sample["label_cos"] = torch.tensor(label_cos, dtype=torch.float32).unsqueeze(0)

        return sample

    def __len__(self):
        return self.normal_len + self.emg_len


class Unsupervised(data.Dataset):
    def __init__(self, data, max_conv_size=7):
        # max_conv_size: always set to a odd number

        self.conv_size = max_conv_size

        self.speed = np.array(data["Speed"], dtype=float)
        self.cos = np.cos(np.array(data["Bearing"], dtype=float))
        self.sin = np.sin(np.array(data["Bearing"], dtype=float))

        self.len = len(self.speed)

        zeros = np.zeros((self.conv_size // 2))

        self.speed = np.concatenate((np.concatenate((zeros, self.speed), axis=0), zeros), axis=0)
        self.cos = np.concatenate((np.concatenate((zeros, self.cos), axis=0), zeros), axis=0)
        self.sin = np.concatenate((np.concatenate((zeros, self.sin), axis=0), zeros), axis=0)

    def __getitem__(self, index):
        save_index = index

        index += self.conv_size // 2

        sample = {}

        sample["index"] = index
        sample["save_index"] = save_index

        sample["train_speed"] = torch.tensor(
            self.speed[index - self.conv_size // 2 : index + self.conv_size // 2 + 1], dtype=torch.float32
        ).unsqueeze(0)
        sample["train_sin"] = torch.tensor(self.sin[index - self.conv_size // 2 : index + self.conv_size // 2 + 1], dtype=torch.float32).unsqueeze(0)
        sample["train_cos"] = torch.tensor(self.cos[index - self.conv_size // 2 : index + self.conv_size // 2 + 1], dtype=torch.float32).unsqueeze(0)

        return sample

    def __len__(self):
        return self.len
