import numpy as np
import cv2 as cv


def detect_accidents(normal_data, emg_data, accident_th):
    """detects accidents according to the functions values and the accident_th.
    the normal data is used as calibration since the emergency data wouldn't be available
    in real world.

    Args:
        normal_data: dict with normal data
        emg_data: dict with emergency data
        accident_th: threshold value

    Returns:
        func: function values
        index: index of the detected accidents
    """

    normal_speed = normal_data["Speed"]
    normal_bearing = normal_data["Bearing"]
    emg_speed = emg_data["Speed"]
    emg_bearing = emg_data["Bearing"]

    speed_count = 0
    speed_diff_mean = 0
    for speed1, speed2 in zip(normal_speed[:-1], normal_speed[1:]):
        diff = np.maximum(speed1 - speed2, 0)
        speed_diff_mean += diff
        speed_count += diff > 0

    speed_diff_mean /= speed_count

    bearing_count = 0
    bearing_diff_mean = 0
    for i in range(1, len(normal_bearing)):
        s1 = normal_speed[i - 1]
        s2 = normal_speed[i]
        b1 = normal_bearing[i - 1]
        b2 = normal_bearing[i]

        if s1 - s2 > 0:
            bearing_count += 1
            bearing_diff_mean += np.abs(b1 - b2)

    bearing_diff_mean /= bearing_count
    print("speed_diff_mean: ", speed_diff_mean)
    print("bearing_diff_mean: ", bearing_diff_mean)

    func = np.zeros_like(emg_speed)
    index = []

    no_accident_count = 5
    for i in range(1, len(emg_speed)):
        s1 = emg_speed[i - 1]
        s2 = emg_speed[i]
        b1 = emg_bearing[i - 1]
        b2 = emg_bearing[i]

        func[i] = (bearing_diff_mean / speed_diff_mean) * (s1 - s2) / speed_diff_mean - np.abs(b1 - b2) / bearing_diff_mean

        no_accident_count += 1

        if func[i] > accident_th and no_accident_count > 5:
            index.append(i)
            no_accident_count = 0

    return func, np.array(index)
