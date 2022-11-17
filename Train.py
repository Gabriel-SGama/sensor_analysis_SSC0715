import torch
import torch.nn as nn

from tqdm import tqdm


def loss_fn(out, label):
    return torch.square(out - label).mean()


def train(model, loader, optimizer):
    """train the model to predict the t+max_conv_size//2 + 1 step.
    This creates relevant features for KMeans, instead of only using random values

    Args:
        model: pytorch model
        loader: train loader
        optimizer: optimizer to update the model
    """
    for e in range(15):
        loss_mean = 0

        for sample in tqdm(loader):
            speed = sample["train_speed"]
            sin = sample["train_sin"]
            cos = sample["train_cos"]

            out = model(speed, sin, cos)
            loss = loss_fn(out["speed"], sample["label_speed"])

            loss_mean += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("loss:", loss_mean / len(loader))
