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
def plotCsv(data):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(data["Speed"], color="b")
    ax2.plot(data["Bearing"], color="r")

    ax1.set_ylabel("Speed (m/s)", color="b")
    ax2.set_ylabel("bearing (°)", color="r")


def drawMap(data, func, accident_th):
    lat_data = np.array(data["Lat"])
    lng_data = np.array(data["Lng"])

    min_lat = np.min(lat_data)
    max_lat = np.max(lat_data)

    min_lng = np.min(lng_data)
    max_lng = np.max(lng_data)

    lat_scale = (_MAP_HEIGHT - 1) / (max_lat - min_lat)
    lng_scale = (_MAP_LENGTH - 1) / (max_lng - min_lng)

    map_img = np.zeros((_MAP_HEIGHT, _MAP_LENGTH, 3), np.uint8)

    func = np.maximum(0, func)

    max_func = np.max(func)
    min_func = np.min(func)

    hue_min = 60
    hue_max = 180
    hue_diff = hue_max - hue_min

    func_scale = 1 / (max_func - min_func)
    saturation = 255
    value = 255

    for i, (lat, lng) in enumerate(zip(lat_data, lng_data)):
        colunm = round((lat - min_lat) * lat_scale)
        line = round((lng - min_lng) * lng_scale)

        color = (hue_min + hue_diff * func[i - 1] * func_scale if i > 0 else hue_min, saturation, value)

        cv.circle(map_img, (colunm, line), 3, color, -1)

        if func[i - 1] > accident_th:
            cv.circle(map_img, (colunm, line), 15, (hue_max, saturation, value), 2)

    map_img = cv.cvtColor(map_img, cv.COLOR_HSV2RGB)

    plt.figure()
    plt.imshow(map_img)
