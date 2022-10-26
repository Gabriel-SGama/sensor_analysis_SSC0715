import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
import logic

root_normal_data = "Dados-GPS-e-IMU-Normal"
root_emg_data = "Dados-GPS-e-IMU-Emergency-Stop"

_ACCIDENT_TH = 30
_BUMP_TH = 2  # 3x mean value
_EMG_START_ITER = 5850
_EMG_END_ITER = 43270


if __name__ == "__main__":
    # -------------READ-------------
    # -------------NORMAL-------------
    normal_csv_path = "Dados-GPS-e-IMU-Nornal/2022-10-17_154353_normal1.csv"
    normal_csv = pd.read_csv(normal_csv_path)

    normal_acc_txt_path = "Dados-GPS-e-IMU-Nornal/Sensors_normal_accelerometer_t1.txt"
    normal_acc_txt_values = utils.readTxt(normal_acc_txt_path, _EMG_START_ITER, _EMG_END_ITER)

    # -------------EMERGENCY-------------
    emg_csv_path = "Dados-GPS-e-IMU-Emergency-Stop/C2_2022-10-17_160507_Sao_Carlos.csv"
    emg_csv = pd.read_csv(emg_csv_path)

    emg_acc_txt_path = "Dados-GPS-e-IMU-Emergency-Stop/Sensors_c2_accelerometer_t1.txt"
    emg_acc_txt_values = utils.readTxt(emg_acc_txt_path, _EMG_START_ITER, _EMG_END_ITER)

    emg_gyro_txt_path = "Dados-GPS-e-IMU-Emergency-Stop/Sensors_c2_gyroscope_t1.txt"
    emg_gyro_txt_values = utils.readTxt(emg_gyro_txt_path, _EMG_START_ITER, _EMG_END_ITER)

    # -------------PROCESS-------------

    # -------------ACCIDENTS-------------
    func_normal, normal_accident_indexs = logic.detect_accidents(normal_csv, normal_csv, _ACCIDENT_TH)
    func_emg, emg_accident_indexs = logic.detect_accidents(normal_csv, emg_csv, _ACCIDENT_TH)

    # -------------PLOT-------------
    # -------------DETECT-------------
    utils.plotFunc(func_normal, normal_accident_indexs, "normal function values")
    utils.plotFunc(func_emg, emg_accident_indexs, "emergency function values")

    # -------------NORMAL-------------
    utils.plotCsv(normal_csv, "normal data", normal_accident_indexs)
    utils.drawMap(normal_csv, func_normal, normal_acc_txt_values, normal_accident_indexs, _BUMP_TH, "normal map")

    plt.figure()
    plt.title("normal accelerometer data")
    plt.xlabel("step")
    plt.ylabel("m/s^2")
    plt.plot(normal_acc_txt_values[:, 1:], label=["X", "Y", "Z"])
    plt.legend()

    # -------------EMERGENCY-------------
    utils.plotCsv(emg_csv, "emergency data", emg_accident_indexs)
    utils.drawMap(emg_csv, func_emg, emg_acc_txt_values, emg_accident_indexs, _BUMP_TH, "emergency map")

    plt.figure()
    plt.title("emergency accelerometer data")
    plt.xlabel("step")
    plt.ylabel("m/s^2")
    plt.plot(emg_acc_txt_values[:, 1:], label=["X", "Y", "Z"])
    plt.legend()

    plt.show()
