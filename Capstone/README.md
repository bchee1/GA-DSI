# Classifying Chest X-Rays Using Various CNN Architectures

## Background
During the COVID-19 pandemic, healthcare professionals were pressed to diagnose and treat patients with severe pneumonia in high numbers that stretched hospital bed capacity. Chest X-rays became one of the diagnostic tools to help in confirming such a diagnosis, and formed large corpa of data. 

One such dataset is the BIMCV-COVID-19
https://github.com/BIMCV-CSUSP/BIMCV-COVID-19

The dataset has over 20000 images from patients who were either suspected of, or had pneumonia over the course of their visit

## Table of Contents
Problem Statement
Dataset
Exploratory Data Analysis
Preprocessing
Model (Convolutional Neural Network)
References

## Problem Statement
Train a useful convolutional neural network to detect and classify pneumonia in patients.

## Dataset
The BIMCV-COVID-19 dataset consists of both images and structured data.

The image dataset consists of 20000+ images, which consist of 13114 patients.
All images are originally 2823 x 2593 pixels, but have been resized to 524x524 pixels.

Additionally, structured data is also given to us for each image, including features such as age of patient, gender, diagnoses of disease, machine data for the x-rays and so forth.

## Exploratory Data Analysis
Most of the features are based on how the X-rays were taken, with regard to the projection of the images, types of machines used, exposure time etc.

The average age is 63 years old, with the gender split being about roughly even. Target variable split of patients experiencing pneumonia was also 0.25.

## Scripts

Description	|Script	
---	|---	
Data Generator for test/train/validation sets|pneumo_data_generator.py

Data link for trained model and images can be found at:
https://drive.google.com/drive/folders/1y0uSCpKi06vf-EbmnIjeGIhTRHb54ESD?usp=sharing

### Preprocessing

First, the label 'Group' was one hot encoded to reflect only, as opposed to the hierarchical categorical labels in the original data set. None of the other features were taken into the CNN nets other than the images themselves, as well as the target variable.

## CNN Model
4 CNN models were explored
- CNN with 3 convolution layers
- VGG16 with Dropout
- ResNet50 with Dropout
- InceptionNet with Dropout

Optimizer used was Adam with a learning rate of 0.01. The main metric used for model tuning and evaluation was accuracy, due to the relative balanced nature of the classes.


## Results (Convolutional Neural Network)
|Model	|Train Score|Validation Score|Test Score|No. of Params| Training Time|
|---	|---	|---|---|---|---|
|CNN|0.8044|0.8111|0.7930|60,940,898|17min|
|VGG16|0.8044|0.8111|0.7962|15,238,018|32 min|
|ResNet50|0.8041|0.8111|0.8050|25,678,786|47 min|
|InceptionNet|0.8042 |0.8111|0.8045|55,221,090|1h|

## References
https://github.com/BIMCV-CSUSP/BIMCV-COVID-19
https://cv-tricks.com/cnn/understand-resnet-alexnet-vgg-inception/