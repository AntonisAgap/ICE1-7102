import keras
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt

# functions declaration
def load_cfar10_batch(file):
    with open(file, mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels
#variables declaration
batch_size = 64
epochs = 25
num_classes = 10
#loading training data
X_train, y_train = load_cfar10_batch('data_batch_3')
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
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
# compile the model
model.compile(optimizer=keras.optimizers.Adadelta(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
print('CNN compiled successfully')

# fit model parameters, using the train set for training and validation set for validation
history = model.fit(X_train, y_train,
          batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(X_validation, y_validation))
print('CNN trained successfully')
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#saving trained model
model_name = 'CNN_cifar_10_model.h5'
model.save(model_name)
print('Model saved successfully')
