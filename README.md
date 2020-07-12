# Optical-Flow-Shape-Dataset

## Code to create data set for optical flow tasks on fly

This code can be used to quickly generate optical flow data and test your models. To approximate the effectiveness of the model on more complex real world data you can run `data_shape_2.py` which creates dataset with greater occlusion.

Each datapoint consist of two images with a randomly generated shape imposed on a black background and its calculated optical flow.

`data_shape.py` is used to create datapoints where a single shape is present

![alt text]()

`data_shape_double.py` is used to create datapoints where two shapes are present in each image


