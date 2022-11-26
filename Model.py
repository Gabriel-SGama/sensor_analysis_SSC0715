import torch
import torch.nn as nn


class Traj_enc(nn.Module):
    def __init__(self):
        super(Traj_enc, self).__init__()

        self.relu = nn.ReLU()
        self.speed_c1 = nn.Conv1d(1, 10, kernel_size=5)
        self.speed_bn1 = nn.BatchNorm1d(10)

        self.sin_c1 = nn.Conv1d(1, 10, kernel_size=5)
        self.sin_bn1 = nn.BatchNorm1d(10)

        self.cos_c1 = nn.Conv1d(1, 10, kernel_size=5)
        self.cos_bn1 = nn.BatchNorm1d(10)

        self.combined_c = nn.Conv1d(30, 40, kernel_size=3)
        self.combined_bn = nn.BatchNorm1d(40)

        self.enc_fc = nn.Linear(40, 10)
        self.out_bn = nn.BatchNorm1d(10)

    def forward(self, speed, sin, cos):
        speed = self.speed_c1(speed)
        speed = self.speed_bn1(speed)
        speed = self.relu(speed)

        sin = self.sin_c1(sin)
        sin = self.sin_bn1(sin)
        sin = self.relu(sin)

        cos = self.cos_c1(cos)
        cos = self.cos_bn1(cos)
        cos = self.relu(cos)

        combined = torch.cat([speed, sin, cos], dim=1)

        combined = self.combined_c(combined)
        combined = self.combined_bn(combined)
        combined = self.relu(combined)

        combined = combined.view(-1, 40)
        out = self.enc_fc(combined)
        out = self.out_bn(out)
        # out = self.relu(out)

        return out


class Traj_model(nn.Module):
    def __init__(self):
        super(Traj_model, self).__init__()

        self.enc = Traj_enc()
        self.speed_dec_fc1 = nn.Linear(10, 1)

    def forward(self, speed, sin, cos, return_enc=False):
        enc = self.enc(speed, sin, cos)

        speed = self.speed_dec_fc1(enc)

        pred = {}
        pred["speed"] = speed

        if return_enc:
            return enc, pred

        return pred


class class_model(nn.Module):
    def __init__(self, pretrained_model):
        super(class_model, self).__init__()

        self.enc = pretrained_model.enc
        self.class_dec = nn.Linear(10, 1)

    def forward(self, speed, sin, cos):
        out = self.enc(speed, sin, cos)
        out = self.class_dec(out)
        out = torch.sigmoid(out)

        return out
