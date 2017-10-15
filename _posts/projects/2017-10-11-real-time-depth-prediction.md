---
title:  "Real-time Depth Prediction"
date:   2017-10-11 23:05:33 -0400
thumbnail_url: /images/real-time_depth_prediction/real_time_depth_net_screenshot.png
description: "Real-time depth prediction from webcam using deep learning.  RGB-depth image pairs were collected automatically using a robot equipped with a 3D sensor."
---

## Contents
1. Overview
2. Hardware
3. Data collection
4. Data preprocessing
5. Model training
7. Robotic deployment
8. Future improvements

The full code can be found here:

## Overview
Objectives:
1. Automated data collection
2. Real-time neural network deployment

The goal of this project was to achieve real-time depth vision on a robot through a completely "in-house, start-to-finish" procedure.  By "in-house, start-to-finish", I mean that this project included everything from data collection through model deployment.

Single image depth prediction involves mapping each pixel of an RGB image to a single continuous depth value. This can be thought of as an image segmentation problem except that it is a regression rather than a classification.  

Constraints:
- Depth predictions are mapped from a single image
- Model must be deployed locally on an embedded system
- Inference in real-time (I define real-time as >10 FPS)

## Hardware
For the robot, I used a [Turtlebot 2](http://www.turtlebot.com/turtlebot2/) equipped with an [ASUS Xtion Pro](https://www.asus.com/us/3D-Sensor/Xtion_PRO_LIVE/) 3D sensor and a [Jetson TX1](https://developer.nvidia.com/embedded/buy/jetson-tx1-devkit) embedded board.  

## Data collection
Single image depth prediction involves mapping an RGB image to an equally sized depth map.  In order to train a "mapping" model, I needed to collect RGB-depth image pairs.  The ASUS Xtion Pro 3D sensor has the ability to capture RGB images and depth images at the same but there is an offset calibration required in order to align the RGB and depth images since the physical location of each camera is offset by a few centimeters.  

Another calibration that needs to be done is removing the "fishbowl" effect of the monocular camera.  Since the camera lens is curved, there is a slight curvature in the images that it captures.  [This](http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration) tutorial takes you through the process of calibrating a monocular camera to remove the fishbowl effect.  

Once both of these calibrations are complete, you are able to capture sufficiently aligned RGB-depth image pairs.  However, rather than walking around and capturing the image pairs by hand, I decided to put the robot to work and automatically capture the image pairs by driving the robot around and taking images at 5 second intervals.  The result was a dataset of 8079 640x480 RGB-depth image pairs.

![rgb_set_of_5](/images/real-time_depth_prediction/raw_rgb_example_set.png)
![depth_set_of_5](/images/real-time_depth_prediction/raw_depth_example_set.png)

## Data preprocessing
These image pairs aren't bad but you'll notice that there are is a lot of white space, which represents NAN values.  These NAN values come from two sources: 1) misreadings from the sensor and 2) the alignment shift to match the RGB image.  The sensor has a minimum and maximum distance with which it can measure depths and so objects too close or too far away will be misread.  In order to align the depth map to the RGB image, the whole depth map has to be shifted and the result is that a portion of the edges will have not have any measurements.

Depth images with a significant amount of NAN values, say greater than 25%, I'll just remove.  For the remainder, I'll apply the following algorithm:
1. 2D interpolation to fill the inner NAN values
2. Horizontal and vertical linear interpolations to fill the edges x 2 (applied second time to fill corners)
3. Smooth interpolated values by applying an average pool
4. Fill any remaining NAN values with the mean value of the whole depth image

The result looks something like this:

![raw_depth_image](/images/real-time_depth_prediction/depth_before_preproc.png) ![preprocessed_depth_image](/images/real-time_depth_prediction/depth_after_preproc.png)

The code for this algorithm is linked here:

In addition to this preprocessing, I also apply the following data augmentation methods to greatly increase my data set size:
1. Random scale up to 50% of the original image size
2. Random rotation up to 5 degrees in either direction
3. Random flip about the y-axis
4. Random HSV shift
5. Random contrast multiplier
6. Random brightness multiplier

![before_augmentation_example](/images/real-time_depth_prediction/rgb-depth_example.png)
![after_augmentation_example](/images/real-time_depth_prediction/rgb-depth_augmented_example.png)

## Model training
My model architecture is largely based off of the work of Eigen et al. [1] who used a dual-stack CNN:

![cnn_architecture](/images/real-time_depth_prediction/eigen_depth_cnn_net_architecture.png)

The first CNN stack takes the RGB image as an input and predicts a coarse global-scale depth map at a lower resolution.  The second CNN stack takes both the RGB image and the coarse prediction as inputs and predicts a fine-scaled depth map.  The predictions are scaled up to its original image size using a deconvolutional layer.  

I use the 16-layer VGG net as my coarse network and apply batch norm at every convolutional layer.  I also try adding a mean-variance normalization at the output, which leads to much better stability at the output.  The downside of using an MVN layer is that the outputs don't represent true depth values but rather a relative depth map.  The input images are also scaled down to a 320x240 resolution.  The code for the model is found here:

The model is trained with SGD with a starting learning rate 1e-9 and a batch size of 64.  Here are some example outputs:

![depth_net_results](/images/real-time_depth_prediction/DepthNet_results.png)

## Robotic deployment
To deploy this model in real-time, the model will have to be pruned significantly in order to decrease the computation time per frame.  The easiest way to do this is to lower the resolution of the input images and interpolate the output values back to the original size.  I trained a much slimmer model with an input resolution of 80x60.  Deployed on the TX1, the FPS is still pretty low: [video](https://www.youtube.com/watch?v=odSl6qXdgyM)
[![depth_net_ros](/images/real-time_depth_prediction/real_time_depth_net_screenshot.png)](https://www.youtube.com/watch?v=odSl6qXdgyM)

## Future improvements
The depth predictions shown above are only relative depth maps due to the output normalization layer.  In order to predict true depth values, the model could be better stabilized with extra CNN stacks that predict different levels of coarseness or by simply collecting more data.  

In order to achieve true real-time speeds, more efficient model architectures such as AlexNet or ResNet could be used.  

Overall, this was a fun project and I learned a lot from going through the whole end-to-end process of collecting data and deploying on hardware.  If you have any questions or suggestions please feel free to contact me.

## Citations
[1] D. Eigen, C. Puhrsch, and R. Fergus. Depth Map Prediction from a Single Image using a Multi-Scale Deep Network. In NIPS 2014.
