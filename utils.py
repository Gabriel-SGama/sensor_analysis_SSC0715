import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

_MAP_LENGTH = 1200
_MAP_HEIGHT = 1200

# -------------READ FUNCTIONS-------------
def readTxt(path):
    normal_acc_txt_path = path
    normal_acc_txt_lines = open(normal_acc_txt_path, "r").readlines()
    normal_acc_txt_lines = normal_acc_txt_lines[14:]

    normal_acc_txt_values = np.array([float(value) for line in normal_acc_txt_lines for value in line[:-1].split("\t")])
    normal_acc_txt_values = normal_acc_txt_values.reshape((len(normal_acc_txt_values) // 4, 4))

    return normal_acc_txt_values


# -------------PLOT FUNCTIONS-------------
def plotCsv(data, func=None, accident_th=None):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(data["Speed"], color="b")
    ax2.plot(data["Bearing"], color="g")

    ax1.set_ylabel("Speed (m/s)", color="b")
    ax2.set_ylabel("bearing (°)", color="g")

    if func is not None and accident_th is not None:
        index = np.where(func > accident_th)
        ax1.vlines(
            index,
            ymin=np.min(data["Speed"]),
            ymax=np.max(data["Speed"]),
            color="r",
            linestyles=["dashed"],
            linewidth=2,
            alpha=0.8,
        )


def plotFunc(func, accident_th):
    plt.figure()
    plt.plot(func)

    index = np.where(func > accident_th)
    plt.vlines(
        index,
        ymin=np.min(func),
        ymax=np.max(func),
        color="r",
        linestyles=["dashed"],
        linewidth=2,
        alpha=0.8,
    )


def drawMap(data, func, imu, accident_th, bump_th):
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
    z = imu[:, 3]  # z values
    print("z mean: ", np.mean(z))
    print("z max: ", np.max(z))

    # assumes syncronized values
    index_scale = imu.shape[0] / len(lat_data)

    print("scale:", index_scale)

    # ------PLOT------
    for i, (lat, lng) in enumerate(zip(lat_data, lng_data)):
        colunm = round((lat - min_lat) * lat_scale)
        line = round((lng - min_lng) * lng_scale)

        color = (hue_min + hue_diff * func[i] * func_scale if i > 0 else hue_min, saturation, value)

        cv.circle(map_img, (colunm, line), 3, color, -1)

        if func[i] > accident_th:
            cv.circle(map_img, (colunm, line), 15, (hue_max, saturation, value), 2)

        for j in range(int(index_scale)):
            if z[round(i * index_scale + j)] > bump_th:
                cv.circle(map_img, (colunm, line), 15, (90, saturation, value), 2)
                break

    map_img = cv.cvtColor(map_img, cv.COLOR_HSV2RGB)

    plt.figure()
    plt.imshow(map_img)
