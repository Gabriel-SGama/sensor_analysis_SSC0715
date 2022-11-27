from sklearn.cluster import KMeans
import torch
import numpy as np
import cv2 as cv


def classify_supervised(model, loader, max_conv_size, device):
    model.eval()

    accident_index = []

    with torch.no_grad():
        for i, sample in enumerate(loader):
            speed = sample["train_speed"].to(device)
            sin = sample["train_sin"].to(device)
            cos = sample["train_cos"].to(device)

            out = model(speed, sin, cos)

            if out > 0.5:
                accident_index.append(i)

    return accident_index


def extract_features(model, loader, max_conv_size, device):
    """given a pre trained model, this function extract the predicted speed
    and extracted features.


    Args:
        model: pre trained model
        loader: test loader
        max_conv_size: time window

    Returns:
        enc_features: list with the extract features from every time step
        pred: dict with the speed prediction (maybe also prediction the cos and sin might help)
    """

    model.eval()

    speed_pred = []
    enc_features = []

    pred = {}

    with torch.no_grad():
        for sample in loader:
            speed = sample["train_speed"].to(device)
            sin = sample["train_sin"].to(device)
            cos = sample["train_cos"].to(device)

            enc, out = model(speed, sin, cos, return_enc=True)

            enc_features.append(enc.cpu().squeeze().numpy())
            speed_pred.append(out["speed"].cpu().squeeze().item())

    pred["speed"] = np.concatenate((np.zeros(max_conv_size // 2 + 1), np.array(speed_pred)[: -max_conv_size // 2]))

    return enc_features, pred


def cluster(normal_enc_features, emg_enc_features):
    """using the features extract with the pre trained model,
    the KMeans cluster each time step.


    Args:
        normal_enc_features: list with normal data features
        emg_enc_features: list with emergency data features

    Returns:
        normal_labels: KMeans labels for normal data
        emg_labels: KMeans labels for emergency data
        normal_dist: KMeans dist to the class centroid for normal data
        emg_dist: KMeans dist to the class centroid for emergency data
    """

    features = np.concatenate((normal_enc_features, emg_enc_features))
    kmeans = KMeans(n_clusters=3, random_state=0).fit(features)

    normal_labels = kmeans.labels_[: len(normal_enc_features)]
    emg_labels = kmeans.labels_[len(normal_enc_features) :]

    centers = kmeans.cluster_centers_

    normal_dist = np.zeros(len(normal_enc_features))
    emg_dist = np.zeros(len(emg_enc_features))

    for i, feat in enumerate(normal_enc_features):
        normal_dist[i] = np.square(centers[normal_labels[i]] - feat).mean()

    for i, feat in enumerate(emg_enc_features):
        emg_dist[i] = np.square(centers[emg_labels[i]] - feat).mean()

    return normal_labels, emg_labels, normal_dist, emg_dist
