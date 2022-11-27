import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

import torch

import Utils
import Logic
import Dataloader
import Model
import Train
import Classify


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
    supervised_dataset = Dataloader.Supervised(emg_csv, emg_accident_indexs, max_conv_size=max_conv_size)
    normal_unsupervised = Dataloader.Unsupervised(normal_csv)
    emg_unsupervised = Dataloader.Unsupervised(emg_csv)

    unsupervised_train_loader = torch.utils.data.DataLoader(trajectory_dataset, batch_size=_BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    supervised_train_loader = torch.utils.data.DataLoader(supervised_dataset, batch_size=_BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    normal_test_loader = torch.utils.data.DataLoader(normal_unsupervised, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
    emg_test_loader = torch.utils.data.DataLoader(emg_unsupervised, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    traj_model = Model.Traj_model()
    optimizer = torch.optim.Adam(traj_model.parameters(), lr=0.001)

    Train.train_unsupervised(traj_model, unsupervised_train_loader, optimizer, device)
    traj_model_unsupervised = copy.deepcopy(traj_model)
    class_model = Train.train_supervised(traj_model, supervised_train_loader, device)
    normal_accident_indexs_supervised = Classify.classify_supervised(class_model, normal_test_loader, max_conv_size, device)
    emg_accident_indexs_supervised = Classify.classify_supervised(class_model, emg_test_loader, max_conv_size, device)

    # -------------CLUSTERING-------------
    normal_enc_features, normal_pred = Classify.extract_features(traj_model_unsupervised, normal_test_loader, max_conv_size, device)
    emg_enc_features, emg_pred = Classify.extract_features(traj_model_unsupervised, emg_test_loader, max_conv_size, device)

    normal_labels, emg_labels, normal_dist, emg_dist = Classify.cluster(normal_enc_features, emg_enc_features)

    # -------------PLOT-------------
    # -------------NORMAL-------------
    Utils.plotCsv(
        normal_csv, "Normal data", pred=normal_pred, labels=normal_labels, dist=normal_dist, accident_indexs=normal_accident_indexs_supervised
    )

    # -------------EMERGENCY-------------
    Utils.plotCsv(emg_csv, "Emergency data", pred=emg_pred, labels=emg_labels, dist=emg_dist, accident_indexs=emg_accident_indexs_supervised)

    # -------------GROUND TRUTH-------------
    Utils.plotGT(normal_csv, emg_csv, "Ground Truth", normal_accident_indexs, emg_accident_indexs)

    plt.show()
