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
<p align="center">
  <img src="imgs/normal_func.png">
</p>

<p align="center">
  <img src="imgs/emg_func.png">
</p>

<p align="center">
  <img src="imgs/emg_map.png">
</p>

## Unsupervised approach
```
python main.py
```

A 1D convolutional neural network is trained to predict the next speed value to generate relevant features. After that, the method KNN-means is applied to the encoder features to classify each sequence step.

The "accidents" are marked according the Handmade approach.

#### Cluster example
<p align="center">
  <img src="imgs/normal_uns.png">
</p>

<p align="center">
  <img src="imgs/emg_uns.png">
</p>
