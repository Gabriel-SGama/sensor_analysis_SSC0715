from sklearn.cluster import KMeans
import torch
import numpy as np
import cv2 as cv


def extract_features(model, loader, max_conv_size):

    model.eval()

    speed_pred = []
    enc_features = []

    pred = {}

    with torch.no_grad():
        for sample in loader:
            speed = sample["train_speed"]
            sin = sample["train_sin"]
            cos = sample["train_cos"]

            enc, out = model(speed, sin, cos, return_enc=True)

            enc_features.append(enc.squeeze().numpy())
            speed_pred.append(out["speed"].squeeze().item())

    pred["speed"] = np.concatenate((np.zeros(max_conv_size // 2 + 1), np.array(speed_pred)[: -max_conv_size // 2]))

    return enc_features, pred


def cluster(normal_enc_features, emg_enc_features):
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
