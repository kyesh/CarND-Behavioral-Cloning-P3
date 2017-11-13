import csv
import cv2
import numpy as np 

correction = 0.2



from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D

model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
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

model.compile(loss='mse',optimizer='adam')

lines = []
with open('data/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		
images = []
measurements = []
for line in lines:
	#Center
	source_path=line[0]
	filename = source_path.split('/')[-1]
	current_path = 'data/data/IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)
	
	#Left
	source_path=line[1]
	filename = source_path.split('/')[-1]
	current_path = 'data/data/IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement + correction)
	
	#right
	source_path=line[2]
	filename = source_path.split('/')[-1]
	current_path = 'data/data/IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement - correction)
	
	

augmented_images, augmented_measurements = [], []
for image,measurment in zip(images,measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurment)
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(measurment*-1.0)
	
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('model.h5')