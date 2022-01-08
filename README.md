## Quickly create toy optical flow datasets on fly to test your deep learning models. ♀️

This code can be used to quickly generate optical flow datasets to test & debug your deep-learning models. 🚴‍♀️

# Why? 👀

If you want to **sanity test** 🧪 your deep learning models & don't want to spend time ⌚ to run one full iteration on the "Flying Chairs" dataset, you can **use this library to generate (easier, smaller & customizable) toy optical flow datasets**. ***If your model fails here, you damn well know it ain't gonna work on "Flying chairs".*** 😭

#### What is an optical flow dataset ❓
  1. Optical flow datasets (like `FLying Chairs`) consist of 3 points. `Image-1`, `Image-2` & an array of shape (Height x Width x 2) which stores the optical flow b/w `Image-1` & `Image-2`

  2. Standard optical flow datasets are `big` & `harder` 🔴 to test with (& rightly so, since these datasets are based on real or close to real life images & scenarios). So why do I need a toy dataset? 🤷‍♂️ 
  
  
  To any one wondering what "flying chair" is? It is a standard dataset that is used to compare performance of optical flow models (https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs). `Think of it as being the GLUE/CIFAR‑100 of optical flow estimation research.`


To approximate the effectiveness of the model on more complex real world data you can run `data_shapes_double.py` which creates dataset with greater occlusion.

Each datapoint consist of two images with a randomly generated shape imposed on a black background and its calculated optical flow.

`data_shapes.py` is used to create datapoints where a single shape is present

![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20single/0_gt.png)
![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20single/0_img1.png)
![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20single/0_img2.png)

![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20single/385_gt.png)
![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20single/385_img1.png)
![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20single/385_img2.png)

`data_shapes_double.py` is used to create datapoints where two shapes are present in each image

![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20double/32_flow0.png)
![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20double/32_img1.png)
![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20double/32_img2.png)

![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20double/460_flow0.png)
![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20double/460_img1.png)
![alt text](https://github.com/RahulSChand/Optical-Flow-Shape-Dataset/blob/master/sample%20data%20double/460_img2.png)



