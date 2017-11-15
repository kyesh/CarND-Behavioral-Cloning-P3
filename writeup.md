# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The train.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I thought the best architectue to start with was one used for Selfdriving cars by Nivida discussed in lectrue.

```
model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3))) #crop
model.add(Lambda(lambda x: (x / 255.0) - 0.5)) #normailize
model.add(Conv2D(24, (5,5), activation="relu", strides=(2,2)))
model.add(Conv2D(36, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(48,(5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(64,(3,3), strides=(2,2), activation="relu"))
#model.add(Conv2D(64,(3,3), strides=(2,2), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```

#### 2. Attempts to reduce overfitting in the model

I found dropouts made the model perform very poorly and thus desided to limit number of epochs to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. I used `model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)` to watch the validation error and make sure the model did not overtrain. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

My driving ablity of the vehicle was quite poor so I thought it was best to stick with the training data provided by udacity as I did not want to traing the vehicle with bad data.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with the NIVIDA Archetecute for selfdriving cars discussed in lectrue because the application seemded to match well. I was have trouble with the last layer of the network throwing an index error so I removed it. I tried more than 10 different varatioins on it all which performed worse. I used additional layers, dropout layers, more nodes. All of which cause the care to drive more in the center of the lane until if simply verred off the road.  

#### 2. Final Model Architecture

```
model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3))) #crop
model.add(Lambda(lambda x: (x / 255.0) - 0.5)) #normailize
model.add(Conv2D(24, (5,5), activation="relu", strides=(2,2)))
model.add(Conv2D(36, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(48,(5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(64,(3,3), strides=(2,2), activation="relu"))
#model.add(Conv2D(64,(3,3), strides=(2,2), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```

#### 3. Creation of the Training Set & Training Process

Because of my poor simulator driving skills I used the training data provided by Udacity for my training data. I agumented this by by flipping steering angle and images to simulate the car driving in the opposite direction as well as using images from the left and right cameras and offsetting the assocaited steering angle. The data was cropped and normalized and the beggining of the network.

# Video Output
https://github.com/kyesh/CarND-Behavioral-Cloning-P3/raw/master/outputVid.mp4
