import numpy as np
import cv2 as cv


def detect_bumps(normal_imu, emg_imu, bump_th):
    normal_z_values = normal_imu[:, 3]  # z values
    emg_z_values = emg_imu[:, 3]  # z values
    normal_z_mean = np.mean(normal_z_values)
    normal_z_var = np.var(normal_z_values)
    print("normal z mean: ", normal_z_mean)
    print("normal z var: ", normal_z_var)
    no_bump_count = 350

    index = []

    for i, emg_z in enumerate(emg_z_values):
        no_bump_count += 1
        # if no_bump_count > 350:
        if no_bump_count > 0:
            if emg_z > 0:
                pdf = 1 / (normal_z_var * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (((emg_z - normal_z_mean) / normal_z_var) ** 2))
                if pdf < bump_th:
                    no_bump_count = 0
                    index.append(i)

    return np.array(index)


def detect_accidents(normal_data, emg_data, accident_th):
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
