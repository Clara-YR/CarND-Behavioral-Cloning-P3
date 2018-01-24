# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./examples/model.png "Loss Line Chart"
[image1]: ./examples/Loss.png "Loss Line Chart"
[image2]: ./examples/loss_screenshot.png "Loss ScreenShot"
[image3]: ./examples/img_center.png "Center Recovery Image"
[image4]: ./examples/img_left.png "Left Recovery Image"
[image5]: ./examples/img_right.png "Right Recovery Image"
[image6]: ./examples/img_center.png "Normal Image"
[image7]: ./examples/img_flip.png "Flipped Image"


**Rubric Points**

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

**Catalog**

- Part 1: Files Submitted & Code Quality
- Part 2: Data Collection and Preprocessing
- Part 3: Model Architecture and Training Strategy
- Part 4: Improvements According to Each Review

---
# Part 1: Files Submitted & Code Quality

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* __model.ipynb__ -- containing the script to create and train the model as well as to plot those images display in this readme
* __model.py__ -- containing the script to create and train the model
* __drive.py__ -- for driving the car in autonomous mode
* __model.h5__ -- containing a trained convolution neural network 
* __README.md__ -- summarizing the results
* __run1.mp4__ -- recording video in autonomous mode
* __examples__ -- folder contains all the images displayed in the README
* __REAND.pdf__ -- the pdf format of this README


### 2. Submission includes functional code

I defined a function `generator(samples, batch_size=32)` (model.ipynb in cell 9 or model.py in line 118-150)

### 3. Submission code is usable and readable

The __model.py__ file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---
# Part 2: Data Collection and Preprocessing

### 1. Data Recording

Only track one was used to collecting and recording data. Training data was chosen to keep the vehicle driving on the road. I used a combination of:

- one lap of clockwise center lane driving 
- one lap of clockwise recovering from the left and right sides of the road 
- counter-clockwise driving

Only track one was used to collecting and record data.

### 2. Appropriate Training & Validation Data

#### 2.1 Using Multiple Cameras

Training data was chosen to capture good driving behavior and keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

Firstly I recorded laps on track one using center lane driving.

Secondly, I recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to steer if it drifts off to the left or the right. These images show what a recovery looks like starting from left to right :

![alt text][image4]
![alt text][image3]
![alt text][image5]

#### 2.2 Data Augmentation

Although I recorded both clockwise and counter-clockwise laps, the left turns in my data are still much more than the right turns, which could contribute to a left turn bias. 

Flipping images is an effective technique for helping with the left turn bias. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

#### 2.3 Data Preprocessing

The data processing is actually carried out at the beginning of the model training.

- __Nomalization__ -- A Lambda layer is used to normalize the data, converting its value range from (0, 255.0) to (-0.5, 0.5). `model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))`

- __Cripping Images__ -- A Cropping layer is used to crip 70 and 25 rows pixels from the top and the bottom of the images respectively. `model.add(Cropping2D(cropping=((70, 25),(0,0))))` 

#### 2.4 Data Split

After the collection process, I had 51264 number of data points. 

I finally randomly shuffled the dataset and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or underfitting.

---
# Part 3: Model Architecture and Training Strategy

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to imply a well built-up CNN and then adapt it to my dataset.

My first step was to use a convolution neural network model similar to the NVIDIA architecture. I thought this model might be appropriate because it is more powerful than LeNet.

**Avoid Underfitting**

My strategy to avoid underfitting is to apply powerful CNN, which is NVIDIA architecture in this project, and collect enough data (more details in Part2: Data Collection and Preprocessing).  

**Avoid Overfitting**

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I two dropout layers in my model architecture. 

**Parameter Tune**

- batch size = 32

- learning rate -- I used an Adam optimizer so that manually training the learning rate wasn't necessary.

- epoch = 5 -- The ideal number of epochs was 5 as evidenced by the Model MSE line chart.
![alt text][image1]


### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer traits:

| Layer         		|     Description	   | 
|:-----------------:|:--------------------:| 
| Input         		| 160x320x3 RGB image  | 
| Lambda Layer    	| Normalization |
| Cropping			| Cropping Images 65x320x3 |
| Convolution_1    	| output depth = 24, kernel size = 5x5, stride = 2x2 |
| RELU_1		       | contained in Convolution_1   			|
| Dropout_1			| drop probability 0.7       |
| Convolution_2		| output depth = 36, kernel size = 5x5, stride = 2x2	|
| RELU_2				| contained in Convolution_2|
| Dropout_2		   | drop probability 0.7  |
| Convolution_3     | output depth = 64, kernel size = 3x3, stride = 1x1|
| RELU_3		    	| contained in Convolution_3  |
| Convolution_4     | output depth = 64, kernel size = 3x3, stride = 1x1|
| RELU_4				| contained in Convolution_4 |
| Convolution_5     |  outputs 84|
| RELU_5				| contained in Convolution_5  |
| Flatten           |   |
| Dense_1             |  outputs 100|
| Dense_2             |  outputs 50|
| Dense_3             |  outputs 10|
| Dense_4             |  outputs 1|

Here is a visualization of the architecture:

![alt text][image0]

### 3. Train Model

I used __model.ipynb__ to train my model and save the final output in __model.h5__. Then I download __model.ipynb__ as __model.py__ and only kept cell [1], [9], [10], [11] and [15] in __model.py__.

At the first time I the model.h5 trained by model.ipynb, but the outcome was not satisfactory. Therefore I collected more data and train my model via model.py, the outcome is as below:

![alt text][image2]

### 4. Run Autonomous Mode and Record Video

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
$ python drive.py model.h5
```
See my outcome in __run1.mp4__.

---

# Part 4: Improvements According to Each Review

#### REVIEW 1

__Q1__: How to visualize the model architecture?

__Solution1__:

I used
 
```
from keras.utils import plot_model
from keras.models import load_model

plot_model(model, to_file='./examples/model.png')
```
to visualize my model architecture.

__Q2__: Q2: My car will lose control after passing the bridge, how can I improve it? 

__Solution2__:

- _More Data_: I collected more turning driving examples. 

- _Aligning the color spaces_: I order to align the color spaces both __drive.py__ and __model.py__ I made changes in __model.py__ as the table below 

| replace | with | to |
|:--------:|:------:|:------:|
|`cv2.imread()`|`PIL.Image.open()`| open or read image|
|`cv2.save()`|`plt.savefig()`|save image|
|`images.append(cv2.flip(image, 1))`|`images.append(np.fliplr(image))`|flip image|


- _Normalization_: I changed my normalization from 
`model.add(Lambda(lambda x: x/225.0 - 0.5, input_shape=(160, 320, 3)))` to 
`
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(160, 320, 3)))`. 

#### REVIEW 2

- __Correction__ was changed from 0.2 to 0.16 (line 36 in model.py)

- __Recovery Data__ I only recorded the process when the car driving from right or left edges to the center of the road instead of the entire lap. To be specific, one lap of recovering from the left and right sides of the road contains control the car to depart and recovery the road center iteratively. The departing driving might confuse the CNN and could be taken as noise. Thus I only record the recovering driving to get rid of such noise.



