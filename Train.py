import torch
import torch.nn as nn

from tqdm import tqdm

import Model


def loss_fn(out, label):
    return torch.square(out - label).mean()


def train_unsupervised(model, loader, optimizer, device):
    """train the model to predict the t+max_conv_size//2 + 1 step.
    This creates relevant features for KMeans, instead of only using random values

    Args:
        model: pytorch model
        loader: train loader
        optimizer: optimizer to update the model
    """

    model.to(device)

    for e in range(20):
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

        print("reg loss:", loss_mean / len(loader))


def train_supervised(pretrained_model, loader, device):
    class_model = Model.class_model(pretrained_model)
    class_model.to(device)
    optimizer = torch.optim.Adam(class_model.parameters())

    class_loss = nn.BCELoss(reduce=False)

    for e in range(10):
        loss_mean = 0

        for sample in tqdm(loader):
            speed = sample["train_speed"].to(device)
            sin = sample["train_sin"].to(device)
            cos = sample["train_cos"].to(device)

            out = class_model(speed, sin, cos)
            accidents = (sample["accident"].to(device) > 0.5).squeeze()
            loss = class_loss(out, sample["accident"].to(device)).squeeze()

            loss += loss * accidents * 5
            loss = loss.mean()
            loss_mean += loss.cpu().item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("class loss:", loss_mean / len(loader))

    return class_model
