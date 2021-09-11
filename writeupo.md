# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./sign1.png "Visualization"
[image15]: ./graphing.png "Training Bar Chart"
[image2]: ./gray.jpg "Grayscaling"
[image3]: ./examples/noise.jpg "Random Noise"
[image4]: ./signs/1.png "Traffic Sign 1"
[image5]: ./signs/2.png "Traffic Sign 2"
[image6]: ./signs/4.png "Traffic Sign 3"
[image7]: ./signs/5.png "Traffic Sign 4"
[image8]: ./signs/6.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 1)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale as the colors of the signs are not really significant in determining what they are. The text and shape make it clear what type of sign it is.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because Neural Networks have a harder time generalizing a range of 0-255. If I normalize the image to 0-1, the model will have better performance.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

Layer (type)                 Shape.                    Parameters
=================================================================
conv2d_29 (Conv2D)           (None, 28, 28, 8)         208       
_________________________________________________________________
activation_48 (Activation)   (None, 28, 28, 8)         0         
_________________________________________________________________
max_pooling2d_18 (MaxPooling (None, 14, 14, 8)         0         
_________________________________________________________________
activation_49 (Activation)   (None, 14, 14, 8)         0         
_________________________________________________________________
conv2d_30 (Conv2D)           (None, 10, 10, 32)        6432      
_________________________________________________________________
activation_50 (Activation)   (None, 10, 10, 32)        0         
_________________________________________________________________
max_pooling2d_19 (MaxPooling (None, 5, 5, 32)          0         
_________________________________________________________________
activation_51 (Activation)   (None, 5, 5, 32)          0         
_________________________________________________________________
conv2d_31 (Conv2D)           (None, 1, 1, 128)         102528    
_________________________________________________________________
flatten_2 (Flatten)          (None, 128)               0         
_________________________________________________________________
activation_52 (Activation)   (None, 128)               0         
_________________________________________________________________
dense_10 (Dense)             (None, 172)               22188     
_________________________________________________________________
activation_53 (Activation)   (None, 172)               0         
_________________________________________________________________
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the model using an Adam optimizer. I didn't set a batch size, so I am assuming that it is equal to 1, I set the number of epochs to 10, and the learning rate to 0.01 (default).

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.94%
* validation set accuracy of 96.03%
* test set accuracy of 93.19%

If a well known architecture was chosen:
* What architecture was chosen? 
I used an implementation of LeNet-5.
* Why did you believe it would be relevant to the traffic sign application?
The traffic sign application didn't really need a large neural network, so I thought that the LeNet-5 would work really well for this application. Initially, I intended to use the LeNet-5 implementation as a base and build from there, but it ended up working really well after I tweaked the number of filters/depth of the Conv2D and Depth layers to the powers of 2.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
It shows that the model is not only just learning the features of the training set and can generalize to new similar data from different sets. It shows that the model is not overfitting.
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The last image may be really hard to classify because the lighting is pretty bad.
#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 Km/H Speed Limit      		| Wild Animal Xing   									| 
| No Clue     			| End of No Passing 										|
| No Vehicles					| No Vehicles											|
| Also No Clue	      		| Right way at next intersection					 				|
| Yield			| Yield      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This relatively low accuracy could be potentially be explained by the wierd images and signs not contained in the dataset.
#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.79), and the image does contain a wild animal crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .79         			| Wild Animal Crossing   									| 
| .11     				| Right Way for the next intersection 										|
| .07					| Slippery Road											|
| .004	      			| Speed limit 20km					 				|
| .003				    | Speed Limit 50km     							|


For the second image, the model is relatively sure that this is a stop sign (probability of 0.79). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .88         			| End of No passing   									| 
| .09    				| Danger curve leftintersection 										|
| .02					| Pedestrians											|
| .002	      			| Ahead only					 				|
| .001				    | Slippy road    							|

For the third image, the model is relatively sure that this is a No vehicles (probability of 0.99). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| No vehicles   									| 
| 1.3 x 10^-5    				| Right way at next intersection 										|
| 8.34 x 10^-6					| 50 km											|
| 4.09 x 10^-6	      			| 30 km					 				|
| 2.22 x 10^-6				    | Keep right    							|

For the fourth image, the model is relatively sure that this is a Right way at next intersection (probability of 0.99). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Right way at next intersection   									| 
| 0.0007    				| 80 km 										|
| 0.0005					| No passing											|
| 0.0005	      			| yield					 				|
| 0.00048				    | Vehicles over 3.5 metric tons    							|

For the five image, the model is relatively sure that this is a yield (probability of 0.99). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Yield   									| 
| 0.0005    				| Right or straight 										|
| 0.0003					| No vehicles											|
| 0.0001	      			| ..ll speed and passing limit					 				|
| 5.58 x 10^-5				    | Keep right    							|






