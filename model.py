import os
import csv


samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


# split the data
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


import cv2
import numpy as np
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # 1. Using Multiple Cameras #
                for i in range(3):
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    angle = float(batch_sample[3])
                    correction = 0.2
                    # left_angle = center_angle + correction
                    if(i==1):
                        angle += correction
                    # right_angle = center_angle - correction
                    elif(i==2):
                        angle -= correction
                    images.append(image)
                    angles.append(angle)

                    # 2. DATA AUGMENTATION #
                    images.append(cv2.flip(image, 1))
                    angles.append(angle*-1.0)

            # convert data to NumPy -- the format Keras requires
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# use the generator functiont to compile and train the model
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


# NIVIDIA Architecture
model = Sequential()
# 3. DATA NORMALIZATION
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
# 4. CROPPING IMAGES
model.add(Cropping2D(cropping=((70, 25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(0.7))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(0.7))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit_generator(train_generator, 
					samples_per_epoch=len(train_samples)*6, 
					validation_data=validation_generator,
					nb_val_samples=len(validation_samples)*6, 
					nb_epoch=5, verbose=1)

model.save('model.h5')
print('The model has been saved successfully.')
