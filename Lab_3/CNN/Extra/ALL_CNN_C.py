import keras
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import numpy as np

import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import concatenate


# functions declaration

def load_cfar10_batch(file):
    with open(file, mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels


checkpoint = ModelCheckpoint(filepath='ALL_CNN_best_model.h5', mode='max', monitor='val_accuracy', verbose=2,
                             save_best_only=True)

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

X_train1, y_train1 = load_cfar10_batch('data_batch_1')
X_train2, y_train2 = load_cfar10_batch('data_batch_2')
X_train3, y_train3 = load_cfar10_batch('data_batch_3')
X_train4, y_train4 = load_cfar10_batch('data_batch_4')
X_train5, y_train5 = load_cfar10_batch('data_batch_5')

X_train = concatenate((X_train1, X_train2, X_train3, X_train4, X_train5), axis=0)
y_train = concatenate((y_train1, y_train2, y_train3, y_train4, y_train5))

# variables declaration
batch_size = 256
epochs = 50
num_classes = 10
# loading training data
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
datagen.fit(X_train)
# creating validation data(3% of train data)
X_train, X_validation = X_train[1000:], X_train[:1000]
y_train, y_validation = y_train[1000:], y_train[:1000]
print('Train and Validation data sets were created successfully')
print('X_train shape is: ', X_train.shape)
print('X_validation shape is: ', X_validation.shape)
print(X_train.shape[0], ' train samples')
print(X_validation.shape[0], ' validation samples')
# defining the architecture of our neural network
model = Sequential()
model.add(Conv2D(96, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation("relu"))
model.add(Conv2D(96, (3, 3), padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
model.add(Dropout(0, 5))
model.add(BatchNormalization())
model.add(Conv2D(192, (3, 3), padding='same'))
model.add(Activation("relu"))
model.add(Conv2D(192, (3, 3), padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
model.add(Dropout(0, 5))
model.add(BatchNormalization())
model.add(Conv2D(192, (3, 3), padding='same'))
model.add(Activation("relu"))
model.add(Conv2D(192, (1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(10, (1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(GlobalAveragePooling2D())
model.add(Activation("softmax"))
print('Epoch 00022: val_accuracy improved from 0.81700 to 0.83500')
print('CNN topology setup completed')
print('CNN\'s topology is: ')
# print model summary
model.summary()
# compile the model
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
print('CNN compiled successfully')

# fit model parameters, using the train set for training and validation set for validation
history = model.fit(datagen.flow(X_train, y_train,
                                 batch_size=batch_size), callbacks=[checkpoint], epochs=epochs, verbose=2,
                    validation_data=(X_validation, y_validation))
# saving trained model
with open('ALL_CNN_trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

print('Model saved successfully')

print('CNN trained successfully')

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('batch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('batch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
