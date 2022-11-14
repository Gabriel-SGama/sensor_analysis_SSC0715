# sensor_analysis_SSC0715

This is a repository for my project for SSC0715. Sensor analysis to detect variation in the sensor data (IMU and GPS were used)


## Handmade approach

```
cd handmade_method
python plot.py
```

function used to detect accidents:
$$ f = \left( \frac{bearingDiffMean}{speedDiffMean}\right) \frac{s_1-s_2}{speedDiffMean} - \frac{|b_1-b_2|}{bearingDiffMean} $$


The core idea is to weight the velocity term and discount when the car is turning, as it is naturally reducing the velocity


If the function value is higher than a certain threshold, it is considered an accident.

#### Function values and map
![normal_func](imgs/normal_func.png?raw=true "normal function")
![emg_func](imgs/emg_func.png?raw=true "emergency function")
![emg_map](imgs/emg_map.png?raw=true "emergency map")

## Unsupervised approach
```
python main.py
```

A 1D convolutional neural network is trained to predict the next speed value to generate relevant features. After that, the method KNN-means is applied to the encoder features to classify each sequence step.

The "accidents" are marked according the Handmade approach.

#### Cluster example
![normal_uns](imgs/normal_uns.png?raw=true "normal cluster")
![emg_uns](imgs/emg_uns.png?raw=true "emg cluster")
