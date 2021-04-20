import keras
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import itertools

# functions declaration
def load_cfar10_batch(file):
    with open(file, mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels

def plot_confusion_matrix(cm, classes,normalize=False):
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# loading test data
X_test, y_test = load_cfar10_batch('test_batch')

loaded_model = keras.models.load_model('CNN_cifar_10_model.h5.h5')
print("Model Loaded Successfully")

y_test_predictions_vectorized = loaded_model.predict(X_test)
y_test_predictions = np.argmax(y_test_predictions_vectorized, axis=1)

class_id = 0
class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
while (sum(y_test_predictions == class_id) > 4):
    tmp_idxs_to_use = np.where(y_test_predictions == class_id)

    # create new plot window
    plt.figure()

    # plot 4 images as gray scale
    plt.subplot(221)
    plt.imshow(X_test[tmp_idxs_to_use[0][0]])
    plt.subplot(222)
    plt.imshow(X_test[tmp_idxs_to_use[0][1]])
    plt.subplot(223)
    plt.imshow(X_test[tmp_idxs_to_use[0][2]])
    plt.subplot(224)
    plt.imshow(X_test[tmp_idxs_to_use[0][3]])
    tmp_title = 'Object considered as: ' + class_name[class_id]
    plt.suptitle(tmp_title)

    # show the plot
    plt.show()
    plt.pause(2)

    # update the class to demonstrate index
    class_id = class_id + 1

acc_test = accuracy_score(y_test, y_test_predictions)
pre_test = precision_score(y_test, y_test_predictions, average='macro')
rec_test = recall_score(y_test, y_test_predictions, average='macro')
f1_test = f1_score(y_test, y_test_predictions, average='macro')

print('Accuracy score is: {:.2f}.'.format(acc_test))
print('Precision score is: {:.2f}.'.format(pre_test))
print('Recall score is: {:.2f}.'.format(rec_test))
print('F1 score is: {:.2f}.'.format(f1_test))


conf_matrix = confusion_matrix(y_test, y_test_predictions)

plot_confusion_matrix(conf_matrix, classes= class_name)