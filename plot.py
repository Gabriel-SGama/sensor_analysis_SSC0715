import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
import logic

root_normal_data = "Dados-GPS-e-IMU-Normal"
root_emg_data = "Dados-GPS-e-IMU-Emergency-Stop"

ACCIDENT_TH = 25

if __name__ == "__main__":
    # -------------READ-------------
    # -------------NORMAL-------------
    normal_csv_path = "Dados-GPS-e-IMU-Nornal/2022-10-17_154353_normal1.csv"
    normal_csv = pd.read_csv(normal_csv_path)

    normal_acc_txt_path = "Dados-GPS-e-IMU-Nornal/Sensors_normal_accelerometer_t1.txt"
    normal_acc_txt_values = utils.readTxt(normal_acc_txt_path)

    # -------------EMERGENCY-------------
    emg_csv_path = "Dados-GPS-e-IMU-Emergency-Stop/C2_2022-10-17_160507_Sao_Carlos.csv"
    emg_csv = pd.read_csv(emg_csv_path)

    emg_acc_txt_path = "Dados-GPS-e-IMU-Emergency-Stop/Sensors_c2_accelerometer_t1.txt"
    emg_acc_txt_values = utils.readTxt(emg_acc_txt_path)

    # -------------PROCESS-------------
    func = logic.detect_accidents(normal_csv, emg_csv)

    # -------------PLOT-------------
    # -------------DETECT-------------
    # plt.figure()
    # plt.plot(func)
    utils.plotFunc(func, ACCIDENT_TH)

    # -------------NORMAL-------------
    # utils.plotCsv(normal_csv)

    # plt.figure()
    # plt.plot(normal_acc_txt_values[:, 0], normal_acc_txt_values[:, 1:], label=["X", "Y", "Z"])
    # plt.legend()

    # -------------EMERGENCY-------------
    utils.plotCsv(emg_csv, func, ACCIDENT_TH)
    utils.drawMap(emg_csv, func, emg_acc_txt_values, ACCIDENT_TH, 16.5)

    plt.figure()
    plt.plot(emg_acc_txt_values[:, 0], emg_acc_txt_values[:, 1:], label=["X", "Y", "Z"])
    plt.legend()

    plt.show()
