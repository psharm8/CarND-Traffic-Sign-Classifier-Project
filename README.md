## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/augmented.png "Augmented"
[image4]: ./examples/aug_visualization.png "Visualization"
[image5]: ./signs_from_internet/4_70kmph.jpg "Traffic Sign 1"
[image6]: ./signs_from_internet/12_priority.jpg "Traffic Sign 2"
[image7]: ./signs_from_internet/14_stop.jpg "Traffic Sign 3"
[image8]: ./signs_from_internet/25_roadwork.jpg "Traffic Sign 4"
[image9]: ./signs_from_internet/28_children.jpg "Traffic Sign 5"
[image10]: ./signs_from_internet/33_right_turn.jpg "Traffic Sign 6"
[image11]: ./signs_from_internet/38_keep_right.jpg "Traffic Sign 7"
[image12]: ./examples/predictions.png "Predictions"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/psharm8/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distrubuted over labels

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because traffic signs have high contrast and converting to garyscale does not cause any feature loss. Grayscale conversion is usefull becuase the memory requirement is reduced.

I normalized the image data because it was suggested in the lesson. I don't fully understand the reason but from what I could figure out from further reading is that since we multiply the intensities with weights it is a good practice to have the image normalized so that the resulting weights are not domintaed by very high/low intensities.

Here is an example of few traffic sign image before and after processing (grayscale and normalization).

![alt text][image2]


I decided to generate additional data because there were quite a few lables which had very less number of samples. This could lead to model having a bias towards the labels having very high number of samples.

To add more data to the the data set, I histogram counts and added images to the lables that had less than 700 samples.
I took the images from the same trainig dataset labels and applied a random warp before appending to the training data.
This way there is more variation to the samples rather than having duplicates.

Here is an example of augmented images and the histogram:

![alt text][image3]
![alt text][image4]

The difference between the original data set and the augmented data set is the following

Original Shapes X_train:(34799, 32, 32, 3), y_train:(34799,)    
Processed Shapes X_train:(43880, 32, 32, 1), y_train:(43880,)


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
I used the same LeNet model from the lesson with dropouts in the Fully connected layers.
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x16			|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16		 		|
| Fully connected		| input = 400, outputs = 120        									|
| RELU					|												|
| Dropout |           |
| Fully connected		| input = 120, outputs = 84        									|
| RELU					|												|
| Dropout |           |
| Fully connected		| input = 84, outputs = 43        									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the same Adam Optimizer from LeNet lesson. I set the batch size to be 256 becuase that fits the data within the memory limit of my current GPU. I tried various epochs and noticed that after 30 epochs there was not much improvent in the accuracy. Lowering the learning rate reduced the accuracy so I kept it at 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.960  
* test set accuracy of 0.935

The only change I made to the model was adding dropout layers. Addition of dropout layer if helpful in reducing overfitting of the model.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8]
![alt text][image9] ![alt text][image10] ![alt text][image11]

Among the images found on the internet, the image of stop sign is slightly out of image bounds which could cause it to be misclassified.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)      		| Speed limit (70km/h)   									| 
| Priority road     			| Priority road 										|
| Stop					| General caution											|
| Road work	      		| Road work					 				|
| Children crossing			| Children crossing      							|
| Turn right ahead			| Turn right ahead      							|
| Keep right			| Keep right      							|


The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 85.7%. This is quite low compared to the test accuracy of 93.5%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located at (https://github.com/psharm8/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb#Predict-the-Sign-Type-for-Each-Image)


![alt text][image12]



