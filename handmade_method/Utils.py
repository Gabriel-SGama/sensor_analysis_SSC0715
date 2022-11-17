import colorsys
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

_MAP_LENGTH = 1200
_MAP_HEIGHT = 1200

# -------------READ FUNCTIONS-------------
def readTxt(path, start_index, end_index):
    normal_acc_txt_path = path
    normal_acc_txt_lines = open(normal_acc_txt_path, "r").readlines()
    normal_acc_txt_lines = normal_acc_txt_lines[14:]

    normal_acc_txt_values = np.array([float(value) for line in normal_acc_txt_lines for value in line[:-1].split("\t")])
    normal_acc_txt_values = normal_acc_txt_values.reshape((len(normal_acc_txt_values) // 4, 4))

    return normal_acc_txt_values[start_index:end_index]


# -------------PLOT FUNCTIONS-------------
def plotCsv(data, title, accident_indexs=None):
    fig, ax1 = plt.subplots()
    plt.title(title)
    ax2 = ax1.twinx()
    ax1.plot(data["Speed"], color="b")
    ax2.plot(data["Bearing"], color="g")

    ax1.set_xlabel("step")
    ax1.set_ylabel("Speed (m/s)", color="b")
    ax2.set_ylabel("bearing (°)", color="g")

    if accident_indexs is not None:
        ax1.vlines(
            accident_indexs,
            ymin=np.min(data["Speed"]),
            ymax=np.max(data["Speed"]),
            color="r",
            linestyles=["dashed"],
            linewidth=2,
            alpha=0.8,
        )


def plotTxt(data, title):
    plt.figure()
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("m/s^2")
    plt.plot(data[:, 1:], label=["X", "Y", "Z"])
    plt.legend()


def plotFunc(func, accident_indexs, title):
    plt.figure()
    plt.plot(func)
    plt.title(title)
    plt.xlabel("step")

    plt.vlines(
        accident_indexs,
        ymin=np.min(func),
        ymax=np.max(func),
        color="r",
        linestyles=["dashed"],
        linewidth=2,
        alpha=0.8,
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
