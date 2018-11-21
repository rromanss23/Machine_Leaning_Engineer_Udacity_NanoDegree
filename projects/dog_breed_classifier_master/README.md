# Artificial Intelligence Engineer Nanodegree

[//]: # (Image References)

[image1]: ./images/dog-breed-classifier_screenshot.png "Project Screenshot"


## Project Overview

### Part 1: Image dectector

In this project, we build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images. Given an image, the algorithm will be able to detect if there is a human face in it using OpenCV's implementation of Haar feature-based cascade classifiers. At this point, we need to make the first user experience decision about whether or not notify the user to provide only clear human images, since Haar cascades are good for generic and static objects, and these kind of classifiers are weak detecting variations of that object. A little discussion of this topic is provided in the notebook. 

Right after we proceed to use a pre-trained ResNet-50 model to detect dogs in images using ImageNet. Since this dataset is very large and the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, some computation can be saved by returning the probability vector for the categories related to dog breeds on ImageNet. This way we can build a 'cheap' dog detector in the same way we built a 'human detector' with Haar cascades on the first part. 

### Part 2: Dog breed classifier using Keras 

The next step of the project is to build a dog breed classifier using a deep learning approach getting at least 1% of accuracy. The purpose of this model will be to build an algorithm that given an image of a dog, it identifies an estimate of the canine’s breed, and   if an image of a human is provided, the code will detect if there is a human face in the picture and will return the most resembling dog breed.   

A CNN is architecture using Keras is proposed, and its parameters used, are discussed. The final proposal uses **dropouts** to prevent overfitting, **Global Average Pooling** layers to reduce dimensionality and **Batch Normalization**. In order to improve the accuracy, **Data Augmentation** has been proven as worthy. Once the model was validated relatively quickly, Data augmentation has been demonstrated as an absolute performance enhancer achieving an accuracy of 26.1962% on the test set. 

### Part 3: Transfer learning

On this part, we use [bottleneck features](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) to save time on computation. Since we are working with Keras bottleneck features, that means that we are working with the last activation maps before the fully-connected layers, thus we need to add the fully connected layers. A little experiment and discussion are provided, and InceptionV3 and ResNet-50 comparison are provided. 

### Part 4: Write a custom algorithm

For the last part of this notebook, we build a complete algorithm using the detectors and model written in the previous parts, and we use it to test not previously seen data of dogs and human faces to guess the dog breed or the resembling dog breed. 

![Project Screenshot][image1]


### Install environment, Project instructions and Test

* [Install instructions](https://github.com/udacity/dog-project)
* [Test](http://localhost:8888/notebooks/AIND-DogBreedClassifier/dog_app.ipynb)
* [Demo](https://www.floydhub.com/nvmoyar/projects/dog-breed)

#### Requirements

FloydHub is a platform for training and deploying deep learning models in the cloud. It removes the hassle of launching your own cloud instances and configuring the environment. For example, FloydHub will automatically set up an AWS instance with TensorFlow, the entire Python data science toolkit, and a GPU. Then you can run your scripts or Jupyter notebooks on the instance. 
For this project: 

> floyd run --mode jupyter --gpu --env tensorflow-1.0

You can see your instance on the list is running and has ID XXXXXXXXXXXXXXXXXXXXXX. So you can stop this instance with Floyd stop XXXXXXXXXXXXXXXXXXXXXX. Also, if you want more information about that instance, use Floyd info XXXXXXXXXXXXXXXXXXXXXX

#### Environments

FloydHub comes with a bunch of popular deep learning frameworks such as TensorFlow, Keras, Caffe, Torch, etc. You can specify which framework you want to use by setting the environment. Here's the list of environments FloydHub has available, with more to come!

#### Datasets 

With FloydHub, you are uploading data from your machine to their remote instance. It's a really bad idea to upload large datasets like CIFAR along with your scripts. Instead, you should always download the data on the FloydHub instance instead of uploading it from your machine.

Further Reading: [How and Why mount data to your job](https://docs.floydhub.com/guides/data/mounting_data/)

### Usage 

floyd run --gpu --env tensorflow-1.1 --data nvmoyar/datasets/bottleneck_features/2:bottleneck_features --data nvmoyar/datasets/lfw/1:lfw_ds --data nvmoyar/datasets/dogimages/1:dogimages_ds --mode jupyter

**You only need to mount the data to your job, since datasets have been gently uploaded for you**

#### Output

Often you'll be writing data out, things like TensorFlow checkpoints, updated notebooks, trained models and HDF5 files. You will find all these files, you can get links to the data with:

> floyd output run_ID
