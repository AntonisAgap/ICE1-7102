from keras.models import load_model
from random import randint
from numpy import array
from math import ceil
from math import log10
from numpy import argmax
from numpy import concatenate
from numpy import vstack
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


# variable declaration
n_samples = 1000
n_numbers = 2
largest = 90
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', ' ']
n_chars = len(alphabet)
n_batch = 10

loaded_model =loaded_model = load_model('LSTM_model.h5')
# evaluate on some new patterns
X, y = generate_data(n_samples, n_numbers, largest, alphabet)
result = loaded_model.predict(X, batch_size=n_batch, verbose=0)
# calculate error
expected = [invert(x, alphabet) for x in y]
predicted = [invert(x, alphabet) for x in result]

# show some examples
print('sums:')
for i in range(10):
    print('Expected=%s, Predicted=%s' % (expected[i], predicted[i]))
print('diffr:')
for i in range(10):
    print('Expected=%s, Predicted=%s' % (expected[-i], predicted[-i]))


sequence = str(input("Enter a sequence: "))

# if '+' or '-' in sequence:
#     char_to_int = dict((c, i) for i, c in enumerate(alphabet))
#     sequence_enc = list()
#     for pattern in sequence:
#         integer_encoded = [char_to_int[char] for char in pattern]
#         sequence_enc.append(integer_encoded)
#     Xenc = list()
#     for seq in sequence_enc:
#         pattern = list()
#         for index in seq:
#             vector = [0 for _ in range(len(alphabet))]
#             vector[index] = 1
#             pattern.append(vector)
#         Xenc.append(pattern)
#
#     result = loaded_model.predict(X)
#     result = invert(result, alphabet)