import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
import logic

root_normal_data = "Dados-GPS-e-IMU-Normal"
root_emg_data = "Dados-GPS-e-IMU-Emergency-Stop"

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

    # -------------PROCESS-------------
    func = logic.detect_accidents(normal_csv, emg_csv)

    plt.figure()
    plt.plot(func)

    # -------------PLOT-------------
    # -------------NORMAL-------------
    # utils.plotCsv(normal_csv)

    # plt.figure()
    # plt.plot(normal_acc_txt_values[:, 0], normal_acc_txt_values[:, 1:])

    # -------------EMERGENCY-------------
    utils.plotCsv(emg_csv)
    utils.drawMap(emg_csv, func, 25)

    plt.show()
