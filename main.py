import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

import Utils
import Logic
import Dataloader
import Model
import Train
import Cluster

_BATCH_SIZE = 16
_ACCIDENT_TH = 30

if __name__ == "__main__":
    # -------------READ-------------
    # -------------NORMAL-------------
    normal_csv_path = "Dados-GPS-e-IMU-Normal/2022-10-17_154353_normal1.csv"
    normal_csv = pd.read_csv(normal_csv_path)

    # -------------EMERGENCY-------------
    emg_csv_path = "Dados-GPS-e-IMU-Emergency-Stop/C2_2022-10-17_160507_Sao_Carlos.csv"
    emg_csv = pd.read_csv(emg_csv_path)

    # -------------ACCIDENTS-------------
    func_normal, normal_accident_indexs = Logic.detect_accidents(normal_csv, normal_csv, _ACCIDENT_TH)
    func_emg, emg_accident_indexs = Logic.detect_accidents(normal_csv, emg_csv, _ACCIDENT_TH)

    # -------------TRAIN SPEED AND ANGLE PREDICTOR-------------
    max_conv_size = 7

    trajectory_dataset = Dataloader.Trajectory(normal_csv, emg_csv, max_conv_size)
    normal_unsupervised = Dataloader.Unsupervised(normal_csv)
    emg_unsupervised = Dataloader.Unsupervised(emg_csv)

    train_loader = torch.utils.data.DataLoader(trajectory_dataset, batch_size=_BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    normal_test_loader = torch.utils.data.DataLoader(normal_unsupervised, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
    emg_test_loader = torch.utils.data.DataLoader(emg_unsupervised, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    traj_model = Model.Traj_model()
    optimizer = torch.optim.Adam(traj_model.parameters(), lr=0.001)

    Train.train(traj_model, train_loader, optimizer, device)

    # -------------CLUSTERING-------------
    enc = traj_model.enc

    normal_enc_features, normal_pred = Cluster.extract_features(traj_model, normal_test_loader, max_conv_size, device)
    emg_enc_features, emg_pred = Cluster.extract_features(traj_model, emg_test_loader, max_conv_size, device)

    normal_labels, emg_labels, normal_dist, emg_dist = Cluster.cluster(normal_enc_features, emg_enc_features)

    # -------------PLOT-------------
    # -------------NORMAL-------------
    Utils.plotCsv(normal_csv, "Normal data", pred=normal_pred, labels=normal_labels, dist=normal_dist, accident_indexs=normal_accident_indexs)

    # -------------EMERGENCY-------------
    Utils.plotCsv(emg_csv, "Emergency data", pred=emg_pred, labels=emg_labels, dist=emg_dist, accident_indexs=emg_accident_indexs)
    plt.show()
