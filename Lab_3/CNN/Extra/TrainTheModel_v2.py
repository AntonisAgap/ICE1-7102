import keras
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from ann_visualizer.visualize import ann_viz
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import concatenate
from keras.datasets import cifar10
from keras.callbacks import  ModelCheckpoint


checkpoint = ModelCheckpoint(filepath='MyCNN_best_model.h5',mode='max', monitor='val_accuracy',verbose=2, save_best_only=True)



def load_cfar10_batch(file):
    with open(file, mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)
X_train, y_train = load_cfar10_batch('data_batch_1')

#variables declaration
batch_size = 32
epochs = 50
num_classes = 10

#loading training data



# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
datagen.fit(X_train)
# creating validation data(3% of train data)
X_train, X_validation = X_train[300:], X_train[:300]
y_train, y_validation = y_train[300:], y_train[:300]
print('Train and Validation data sets were created successfully')
print('X_train shape is: ', X_train.shape)
print('X_validation shape is: ', X_validation.shape)
print(X_train.shape[0], ' train samples')
print(X_validation.shape[0], ' validation samples')
#defining the architecture of our neural network
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=X_train.shape[1:]))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
print('CNN topology setup completed')
print('CNN\'s topology is: ')
# print model summary
model.summary()
# compile the dmodel
model.compile(optimizer=keras.optimizers.Adadelta(), loss='categorical_crossentropy', metrics=['accuracy'])
print('CNN compiled successfully')

# fit model parameters, using the train set for training and validation set for validation
history = model.fit(datagen.flow(X_train, y_train,
          batch_size=batch_size) , callbacks = [checkpoint], epochs=epochs, verbose=2 , validation_data=(X_validation, y_validation))
print('CNN trained successfully')

with open('trainHistoryDictv2', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

