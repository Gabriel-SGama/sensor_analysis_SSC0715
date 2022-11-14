import torch
import torch.nn as nn

from tqdm import tqdm


def loss_fn(out, label):
    return torch.square(out - label).mean()


def train(model, loader, optimizer):

    for e in range(15):
        loss_mean = 0

        for sample in tqdm(loader):
            speed = sample["train_speed"]
            sin = sample["train_sin"]
            cos = sample["train_cos"]

            # print("speed: ", speed.shape)

            out = model(speed, sin, cos)
            loss = loss_fn(out["speed"], sample["label_speed"])

            loss_mean += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("loss:", loss_mean / len(loader))
