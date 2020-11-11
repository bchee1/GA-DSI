# Classifying Chest X-Rays Using Various CNN Architectures

## Background
During the COVID-19 pandemic, healthcare professionals were pressed to diagnose and treat patients with severe pneumonia in high numbers that stretched hospital bed capacity. Chest X-rays became one of the diagnostic tools to help in confirming such a diagnosis, and formed large corpa of data. 

One such dataset is the
https://github.com/BIMCV-CSUSP/BIMCV-COVID-19
In October 2017, the National Institute of Health open sourced 112,000+ images of chest chest x-rays. Now known as ChestXray14, this dataset was opened in order to allow clinicians to make better diagnostic decisions for patients with various lung diseases.

## Table of Contents
Objective
Dataset
Exploratory Data Analysis
Pipeline
Preprocessing
Model (Structured Data)
Model (Convolutional Neural Network)
Explanations
References

## Problem Statement
Train a useful convolutional neural network to detect and classify pneumonia in patients.

## Dataset
The ChestXray14 dataset consists of both images and structured data.

The image dataset consists of 112,000+ images, which consist of 30,000 patients. Some patients have multiple scans, which will be taken into consideration. All images are originally 1024 x 1024 pixels.

Due to data sourcing & corruption issues, my image dataset consists of 10,000 of the original 112,000 images. All data is used for the structured model.

Additionally, structured data is also given to us for each image. This dataset includes features such as age, number of follow up visits, AP vs PA scan, and the patient gender.

## Exploratory Data Analysis
When researching the labels, there are 709 original, unique categories present. On further examination, the labels are hierarchical. For example, some labels are only "Emphysema", while others are "Emphysema | Cardiac Issues".

The average age is 58 years old. However, about 400 patients are labeled as months, 1 of them is labeled in days.

## Pipeline
Two pipelines were created for each dataset. Each script is labeled as either "Structured" or "CNN", which indicates which data pipeline the script is part of.

Description	Script	Model
EDA	eda.py	Structured
Resize Images	resize_images.py	CNN
Reconcile Labels	reconcile_labels.py	CNN
Convert Images to Arrays	image_to_array.py	CNN
CNN Model	cnn.py	CNN
Structured Data Model	model.py	Structured
Preprocessing
First, the labels were changed to reflect single categories, as opposed to the hierarchical categorical labels in the original data set. This reduces the number of categories from 709 to 15 categories. The label reduction takes its queue from the Stanford data scientists, who reduced the labels in the same way.

Irrelevant columns were also removed. These columns either had zero variance, or provided minimal information on the patient diagnosis.

Finally, anyone whose age was given in months (M) or days (D) was removed. The amount of data removed is minimal, and does not affect the analysis.

## Model (Structured Data)
The structured data is trained using a gradient boosted classifier. The random forest classifier was also used. When comparing the results, both were nearly equal. The GBM classifier was used due to its speed over the random forest, and due to producing equal or better results to the random forest.

## Results (Structured Data)
Measurement	Score
Model	H2O Gradient Boosting Estimator
Log Loss	1.670
MSE	0.510
RMSE	0.714
R^2	0.967
Mean Per-Class Error	0.933
Model (Convolutional Neural Network)
The CNN was trained using Keras, with the TensorFlow backend.

The model is similar to the VGG architectures; 2 to 3 convolution layers are used in each set of layers, followed by a pooling layer.

Dropout is used in the fully connected layers only, which slightly improved the results.

## Results (Convolutional Neural Network)
Measurement	Score
Accuracy	0.5456
Precision	0.306
Recall	0.553
F1

## References