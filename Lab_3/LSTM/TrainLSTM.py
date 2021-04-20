from random import seed
from random import randint
from numpy import array
from math import ceil
from math import log10
from numpy import argmax
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Masking
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from numpy import concatenate
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae



# function declaration

def diffr(items):
    return items[0] - sum(items[1:])


# generate lists of random integers and their sum
def random_sum_pairs(n_examples, n_numbers, largest):
    X, y = list(), list()
    for i in range(n_examples):
        in_pattern = [randint(1, largest) for _ in range(n_numbers)]
        out_pattern = sum(in_pattern)
        X.append(in_pattern)
        y.append(out_pattern)
    return X, y


def random_diffr_pairs(n_examples, n_numbers, largest):
    X, y = list(), list()
    for i in range(n_examples):
        in_pattern = [randint(1, largest) for _ in range(n_numbers)]
        out_pattern = diffr(in_pattern)
        X.append(in_pattern)
        y.append(out_pattern)
    return X, y


# convert data to strings
def to_string_sum(X, y, n_numbers, largest):
    max_length = n_numbers * ceil(log10(largest + 1)) + n_numbers - 1
    Xstr = list()
    for pattern in X:
        strp = '+'.join([str(n) for n in pattern])
        strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp
        Xstr.append(strp)
    max_length = ceil(log10(n_numbers * (largest + 1)))
    ystr = list()
    for pattern in y:
        strp = str(pattern)
        strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp
        ystr.append(strp)
    return Xstr, ystr


# convert data to strings
def to_string_diffr(X, y, n_numbers, largest):
    max_length = n_numbers * ceil(log10(largest + 1)) + n_numbers - 1
    Xstr = list()
    for pattern in X:
        strp = '-'.join([str(n) for n in pattern])
        strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp
        Xstr.append(strp)
    max_length = ceil(log10(n_numbers * (largest + 1)))
    ystr = list()
    for pattern in y:
        strp = str(pattern)
        strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp
        ystr.append(strp)
    return Xstr, ystr


# integer encode strings
def integer_encode(X, y, alphabet):
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    Xenc = list()
    for pattern in X:
        integer_encoded = [char_to_int[char] for char in pattern]
        Xenc.append(integer_encoded)
    yenc = list()
    for pattern in y:
        integer_encoded = [char_to_int[char] for char in pattern]
        yenc.append(integer_encoded)
    return Xenc, yenc


# one hot encode
def one_hot_encode(X, y, max_int):
    Xenc = list()
    for seq in X:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        Xenc.append(pattern)
    yenc = list()
    for seq in y:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        yenc.append(pattern)
    return Xenc, yenc




def generate_data(n_samples, n_numbers, largest, alphabet):
    # generate pairs
    X_sum, y_sum = random_sum_pairs(n_samples//2, n_numbers, largest)
    X_diffr, y_diffr = random_diffr_pairs(n_samples//2, n_numbers, largest)
    # convert to strings
    X_sum, y_sum = to_string_sum(X_sum, y_sum, n_numbers, largest)
    X_diffr, y_diffr = to_string_diffr(X_diffr,y_diffr, n_numbers, largest)
    # integer encode
    X_sum, y_sum = integer_encode(X_sum, y_sum, alphabet)
    X_diffr, y_diffr = integer_encode(X_diffr, y_diffr, alphabet)
    # one hot encode
    X_sum, y_sum = one_hot_encode(X_sum, y_sum, len(alphabet))
    X_diffr, y_diffr = one_hot_encode(X_diffr, y_diffr, len(alphabet))
    # return as numpy arrays
    X = concatenate((X_sum, X_diffr), axis=0)
    y = concatenate((y_sum, y_diffr))
    X_sum, y_sum = array(X_sum), array(y_sum)
    return X, y

# invert encoding
def invert(seq, alphabet):
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    strings = list()
    for pattern in seq:
        string = int_to_char[argmax(pattern)]
        strings.append(string)
    return ''.join(strings)


# define dataset
seed(1)
n_samples = 10000
n_numbers = 2
largest = 90
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', ' ']
n_chars = len(alphabet)
n_in_seq_length = n_numbers * ceil(log10(largest + 1)) + n_numbers - 1
n_out_seq_length = ceil(log10(n_numbers * (largest + 1)))
# define LSTM configuration
n_batch = 10
n_epoch = 25

# create LSTM
model = Sequential()
model.add(LSTM(100, input_shape=(n_in_seq_length, n_chars)))
model.add(RepeatVector(n_out_seq_length))
model.add(LSTM(50, return_sequences=True))
model.add(TimeDistributed(Dense(n_chars, activation='softmax')))
print('LSTM topology setup completed')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print('LSTM compiled successfully')
# train LSTM
X, y = generate_data(n_samples, n_numbers, largest, alphabet)
print('Training data generated')
history = model.fit(X, y, epochs=n_epoch, batch_size=n_batch)
print('LSTM trained successfully')
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
#saving trained model
model_name = 'LSTM_model.h5'
model.save(model_name)
print('Model saved successfully')

