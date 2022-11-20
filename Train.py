import torch
import torch.nn as nn

from tqdm import tqdm


def loss_fn(out, label):
    return torch.square(out - label).mean()


def train(model, loader, optimizer, device):
    """train the model to predict the t+max_conv_size//2 + 1 step.
    This creates relevant features for KMeans, instead of only using random values

    Args:
        model: pytorch model
        loader: train loader
        optimizer: optimizer to update the model
    """

    model.to(device)

    for e in range(15):
        loss_mean = 0

        for sample in tqdm(loader):
            speed = sample["train_speed"].to(device)
            sin = sample["train_sin"].to(device)
            cos = sample["train_cos"].to(device)

            out = model(speed, sin, cos)
            loss = loss_fn(out["speed"], sample["label_speed"].to(device))

            loss_mean += loss.cpu().item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("loss:", loss_mean / len(loader))
