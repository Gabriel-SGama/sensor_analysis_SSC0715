import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Utils
import Logic

root_normal_data = "Dados-GPS-e-IMU-Normal"
root_emg_data = "Dados-GPS-e-IMU-Emergency-Stop"

_ACCIDENT_TH = 30
_BUMP_TH = 1e-17  # normal distribution probability
_EMG_START_ITER = 5850
_EMG_END_ITER = 43270


if __name__ == "__main__":
    # -------------READ-------------
    # -------------NORMAL-------------
    normal_csv_path = "../Dados-GPS-e-IMU-Normal/2022-10-17_154353_normal1.csv"
    normal_csv = pd.read_csv(normal_csv_path)

    normal_acc_txt_path = "../Dados-GPS-e-IMU-Normal/Sensors_normal_accelerometer_t1.txt"
    normal_acc_txt_values = Utils.readTxt(normal_acc_txt_path, _EMG_START_ITER, _EMG_END_ITER)

    # -------------EMERGENCY-------------
    emg_csv_path = "../Dados-GPS-e-IMU-Emergency-Stop/C2_2022-10-17_160507_Sao_Carlos.csv"
    emg_csv = pd.read_csv(emg_csv_path)

    emg_acc_txt_path = "../Dados-GPS-e-IMU-Emergency-Stop/Sensors_c2_accelerometer_t1.txt"
    emg_acc_txt_values = Utils.readTxt(emg_acc_txt_path, _EMG_START_ITER, _EMG_END_ITER)

    emg_gyro_txt_path = "../Dados-GPS-e-IMU-Emergency-Stop/Sensors_c2_gyroscope_t1.txt"
    emg_gyro_txt_values = Utils.readTxt(emg_gyro_txt_path, _EMG_START_ITER, _EMG_END_ITER)

    # -------------PROCESS-------------
    normal_bumps_index = Logic.detect_bumps(normal_acc_txt_values, normal_acc_txt_values, _BUMP_TH)
    # emg_bumps_index = logic.detect_bumps(normal_acc_txt_values, emg_acc_txt_values, _BUMP_TH)

    # -------------ACCIDENTS-------------
    func_normal, normal_accident_indexs = Logic.detect_accidents(normal_csv, normal_csv, _ACCIDENT_TH)
    func_emg, emg_accident_indexs = Logic.detect_accidents(normal_csv, emg_csv, _ACCIDENT_TH)

    # -------------PLOT-------------
    # -------------FUNCTIONS-------------
    Utils.plotFunc(func_normal, normal_accident_indexs, "normal function values")
    Utils.plotFunc(func_emg, emg_accident_indexs, "emergency function values")

    # -------------NORMAL-------------
    Utils.plotCsv(normal_csv, "normal data", normal_accident_indexs)
    Utils.drawMap(normal_csv, func_normal, normal_acc_txt_values, normal_accident_indexs, normal_bumps_index, "normal map")

    Utils.plotTxt(normal_acc_txt_values, "normal accelerometer data")

    # -------------EMERGENCY-------------
    Utils.plotCsv(emg_csv, "emergency data", emg_accident_indexs)
    Utils.drawMap(emg_csv, func_emg, emg_acc_txt_values, emg_accident_indexs, [], "emergency map")

    Utils.plotTxt(emg_acc_txt_values, "emergency accelerometer data")

    plt.show()
