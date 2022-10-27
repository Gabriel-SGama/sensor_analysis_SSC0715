# sensor_analysis_SSC0715

Repository for a project for SSC0715. Sensor analysis to detect variation in the sensor data (IMU and GPS were used)

```
python plot.py
```
function used to detect accidents:

func = (bearing_diff_mean / speed_diff_mean) * (s1 - s2) / speed_diff_mean - np.abs(b1 - b2) / bearing_diff_mean

The core idea is to weight the velocity term and discount when the car is turning, as it is naturally reducing the velocity