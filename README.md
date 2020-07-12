# Optical-Flow-Shape-Dataset

## Code to create data set for optical flow tasks on fly

This code can be used to quickly generate optical flow data and test your models. To approximate the effectiveness of the model on more complex real world data you can run `data_shape_2.py` which creates dataset with greater occlusion.

Each datapoint consist of two images with a randomly generated shape imposed on a black background and its calculated optical flow.

`data_shape.py` is used to create datapoints where a single shape is present

![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20single/0_gt.png)
![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20single/0_img1.png)
![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20single/0_img2.png)

![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20single/385_gt.png)
![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20single/385_img1.png)
![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20single/385_img2.png)

`data_shape_double.py` is used to create datapoints where two shapes are present in each image

![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20double/32_flow0.png)
![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20double/32_img1.png)
![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20double/32_img2.png)

![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20double/460_flow0.png)
![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20double/460_img1.png)
![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20double/460_img2.png)


