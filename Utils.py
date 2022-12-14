import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import cv2 as cv

_MAP_LENGTH = 1200
_MAP_HEIGHT = 1200

# -------------PLOT FUNCTIONS-------------
def plotCsv(data, title, pred=None, labels=None, dist=None, accident_indexs=None):

    size = 3 if labels is None else 4

    f, ax = plt.subplots(size, 1, figsize=(8, 6))
    f.suptitle(title, fontsize=16)

    f.tight_layout()

    ax[0].set_title("Speed")
    ax[0].plot(data["Speed"], color="b")
    ax[1].set_title("Bearing sin")
    ax[1].plot(np.sin(np.pi * data["Bearing"] / 180.0), color="g")
    ax[2].set_title("Bearing cos")
    ax[2].plot(np.cos(np.pi * data["Bearing"] / 180.0), color="purple")

    if pred is not None:
        ax[0].plot(pred["speed"], color="black")

    if labels is not None:

        colors = cm.nipy_spectral(np.linspace(0, 1, max(labels) + 1))

        ax[3].set_title("Distance")
        ax[3].plot(dist)

        for i in range(len(labels)):
            ax[0].axvspan(i, i + 1, facecolor=colors[labels[i]], alpha=0.4)
            ax[1].axvspan(i, i + 1, facecolor=colors[labels[i]], alpha=0.4)
            ax[2].axvspan(i, i + 1, facecolor=colors[labels[i]], alpha=0.4)
            ax[3].axvspan(i, i + 1, facecolor=colors[labels[i]], alpha=0.4)

    if accident_indexs is not None:
        ax[0].vlines(
            accident_indexs,
            ymin=np.min(data["Speed"]),
            ymax=np.max(data["Speed"]),
            color="r",
            linestyles=["dashed"],
            linewidth=2,
            alpha=0.8,
        )
        # ax[3].vlines(
        #     accident_indexs,
        #     ymin=np.min(dist),
        #     ymax=np.max(dist),
        #     color="r",
        #     linestyles=["dashed"],
        #     linewidth=2,
        #     alpha=0.8,
        # )


def plotGT(normal_data, emg_data, title, normal_accident, emg_accident):

    f, ax = plt.subplots(2, 1, figsize=(8, 6))
    f.suptitle(title, fontsize=16)

    f.tight_layout()

    ax[0].set_title("Normal Speed")
    ax[0].plot(normal_data["Speed"], color="b")
    ax[1].set_title("Emergency Speed")
    ax[1].plot(emg_data["Speed"], color="b")

    if normal_accident is not None:
        ax[0].vlines(
            normal_accident,
            ymin=np.min(normal_data["Speed"]),
            ymax=np.max(normal_data["Speed"]),
            color="r",
            linestyles=["dashed"],
            linewidth=2,
            alpha=0.9,
        )

    if emg_accident is not None:
        ax[1].vlines(
            emg_accident,
            ymin=np.min(emg_data["Speed"]),
            ymax=np.max(emg_data["Speed"]),
            color="r",
            linestyles=["dashed"],
            linewidth=2,
            alpha=0.9,
        )


def drawMap(data, func, imu, accident_indexs, bump_index, title):
    text_pos_x = 0.80

    # ------MAP TRAJECTORY------
    lat_data = np.array(data["Lat"])
    lng_data = np.array(data["Lng"])

    min_lat = np.min(lat_data)
    max_lat = np.max(lat_data)

    min_lng = np.min(lng_data)
    max_lng = np.max(lng_data)

    lat_scale = (_MAP_HEIGHT - 1) / (max_lat - min_lat)
    lng_scale = (_MAP_LENGTH - 1) / (max_lng - min_lng)

    map_img = np.zeros((_MAP_HEIGHT, _MAP_LENGTH, 3), np.uint8)

    # ------FUNC SCALE------
    func = np.maximum(0, func)

    max_func = np.max(func)
    min_func = np.min(func)

    hue_min = 60
    hue_max = 180
    hue_diff = hue_max - hue_min

    func_scale = 1 / (max_func - min_func)
    saturation = 255
    value = 255

    # ------IMU------
    z_values = imu[:, 3]  # z values
    z_mean = np.mean(z_values)
    z_var = np.var(z_values)
    print("z mean: ", z_mean)
    print("z var: ", z_var)

    # assumes synchronized values
    imu_index_scale = len(lat_data) / imu.shape[0]

    print("imu index scale:", imu_index_scale)

    # ------PLOT------
    for i, (lat, lng) in enumerate(zip(lat_data, lng_data)):
        colunm = round((lat - min_lat) * lat_scale)
        line = round((lng - min_lng) * lng_scale)

        color = (hue_min + hue_diff * func[i] * func_scale if i > 0 else hue_min, saturation, value)

        cv.circle(map_img, (colunm, line), 3, color, -1)

    for index in bump_index:
        lat = lat_data[int(index * imu_index_scale)]
        lng = lng_data[int(index * imu_index_scale)]
        colunm = round((lat - min_lat) * lat_scale)
        line = round((lng - min_lng) * lng_scale)

        cv.circle(map_img, (colunm, line), 15, (90, saturation, value), 2)

    for index in accident_indexs:
        lat = lat_data[index]
        lng = lng_data[index]
        colunm = round((lat - min_lat) * lat_scale)
        line = round((lng - min_lng) * lng_scale)

        cv.circle(map_img, (colunm, line), 15, (hue_max, saturation, value), 2)

    cv.circle(map_img, (int(_MAP_LENGTH * text_pos_x), 20), 15, (hue_max, saturation, value), 2)
    cv.circle(map_img, (int(_MAP_LENGTH * text_pos_x), 60), 15, (90, saturation, value), 2)

    map_img = cv.cvtColor(map_img, cv.COLOR_HSV2RGB)

    plt.figure()
    plt.title(title)

    accident_color = colorsys.hsv_to_rgb(1, 1, 1)
    bump_color = colorsys.hsv_to_rgb(90 / hue_max, 1, 1)

    plt.annotate("accident", (int(_MAP_LENGTH * text_pos_x + 20), 20 + 15), color=accident_color)
    plt.annotate("bump", (int(_MAP_LENGTH * text_pos_x + 20), 60 + 15), color=bump_color)
    plt.imshow(map_img)
