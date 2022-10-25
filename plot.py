import pandas as pd
import matplotlib.pyplot as plt


root_normal_data = "Dados-GPS-e-IMU-Normal"
root_emg_data = "Dados-GPS-e-IMU-Emergency-Stop"

if __name__ == "__main__":
    # -------------READ-------------
    # -------------NORMAL-------------
    normal_csv_path = "Dados-GPS-e-IMU-Nornal/2022-10-17_154353_normal1.csv"
    normal_csv_file = pd.read_csv(normal_csv_path)

    # -------------EMERGENCY-------------
    emg_csv_path = "Dados-GPS-e-IMU-Emergency-Stop/C2_2022-10-17_160507_Sao_Carlos.csv"
    emg_csv_file = pd.read_csv(emg_csv_path)

    # -------------PLOT-------------
    # -------------NORMAL-------------
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(normal_csv_file["Speed"], color="b")
    ax2.plot(normal_csv_file["Bearing"], color="r")

    ax1.set_ylabel("Speed (m/s)", color="b")
    ax2.set_ylabel("bearing (°)", color="r")

    # -------------EMERGENCY-------------
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax1.plot(emg_csv_file["Speed"], color="b")
    ax2.plot(emg_csv_file["Bearing"], color="r")

    ax1.set_ylabel("Speed (m/s)", color="b")
    ax2.set_ylabel("bearing (°)", color="r")

    plt.show()
