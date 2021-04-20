# /--------------------------------------/
# NAME: Antonis Agapiou
# DEPARTMENT: INFORMATICS AND COMPUTER ENGINEERING
# E-MAIL: cs141081@uniwa.gr
# A.M: 711141081
# CLASS: PRML
# /--------------------------------------/


# Import Packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, MaxPool2D, UpSampling2D
from keras import backend as K
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.datasets import fashion_mnist
from sklearn import mixture, metrics
import xlsxwriter

# initializing workbook
workbook = xlsxwriter.Workbook("Results.xlsx")
worksheet = workbook.add_worksheet()
col = 0
row = 0
worksheet.write(row, col, 'Cluster')
worksheet.write(row, col + 1, 'Data type')
worksheet.write(row, col + 2, 'Silhouette Coefficient score')
worksheet.write(row, col + 3, 'Calinski-Harabasz score')
worksheet.write(row, col + 4, 'Davies-Bouldin score')
worksheet.write(row, col + 5, 'Homogeneity score')
row = row + 1

# making clusters objects
birch = Birch(n_clusters=10)
kmeans = KMeans(n_clusters=10)
gmm = mixture.GaussianMixture(n_components=10, covariance_type='full')

# making a dictionary with clusters
clustering_algorithms = {
    ('KMeans', kmeans),
    ('Birch', birch),
    ('GM', gmm)
}


# defining functions

# clustering function
def clustering(cluster_name, data):
    clustered_dataset = cluster_name.fit_predict(data)
    return clustered_dataset


# function that writes results to the .xlsx file
def write_excel(cluster_name, data_set, sil_score, ch_score, db_score, hmg_score):
    global worksheet
    global row
    worksheet.write(row, col, cluster_name)
    worksheet.write(row, col + 1, data_set)
    worksheet.write(row, col + 2, '{:.2f}'.format(sil_score))
    worksheet.write(row, col + 3, '{:.2f}'.format(ch_score))
    worksheet.write(row, col + 4, '{:.2f}'.format(db_score))
    worksheet.write(row, col + 5, '{:.2f}'.format(hmg_score))
    row = row + 1


# function that calculates perfomance scores on the clustered data
def perfomance_score(input_values, cluster_indexes, y_true):
    silh_score = metrics.silhouette_score(input_values, cluster_indexes)
    ch_score = metrics.calinski_harabasz_score(input_values, cluster_indexes)
    db_score = metrics.davies_bouldin_score(input_values, cluster_indexes)
    homg_score = metrics.homogeneity_score(y_true, cluster_indexes)
    return silh_score, ch_score, db_score, homg_score


# function tha plots the confusion matrix for each clustered dataset
# and 10 examples of each cluster
def plot_confusion_matrix_clusters_images(y_test, clustered_training_set):
    cm = confusion_matrix(y_test, clustered_training_set)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix", fontsize=30)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Clustering label', fontsize=25)
    plt.show()
    # Plot the actual pictures grouped by clustering
    fig = plt.figure(figsize=(20, 20))
    for r in range(10):
        cluster = cm[r].argmax()
        for c, val in enumerate(X_test[clustered_training_set == cluster][0:10]):
            fig.add_subplot(10, 10, 10 * r + c + 1)
            plt.imshow(val.reshape((28, 28)))
            plt.gray()
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('cluster: ' + str(cluster))
            plt.ylabel('digit: ' + str(r))
    plt.show()


# generating train,test and validation data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

X_train = np.expand_dims(X_train, axis=3)
X_validate = np.expand_dims(X_validate, axis=3)
X_test = np.expand_dims(X_test, axis=3)

# Build the autoencoder
model = Sequential()
model.add(Conv2D(14, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D((2, 2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(7, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D((2, 2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(7, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(14, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(1, kernel_size=3, padding='same', activation='relu'))

model.compile(optimizer='adam', loss="mse")
model.summary()

# Train the model
model.fit(X_train, X_train, epochs=50, batch_size=64, validation_data=(X_validate, X_validate), verbose=2)

# Fitting testing dataset
restored_testing_dataset = model.predict(X_test)

# Observe the reconstructed image quality
plt.figure(figsize=(20, 5))
for i in range(10):
    index = y_test.tolist().index(i)
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_test[index].reshape((28, 28)))
    plt.gray()
    plt.subplot(2, 10, i + 11)
    plt.imshow(restored_testing_dataset[index].reshape((28, 28)))
    plt.gray()

plt.show()

# extracting encoder
encoder = K.function([model.layers[0].input], [model.layers[4].output])
# creating encoded images
encoded_images = encoder([X_test])[0].reshape(-1, 7 * 7 * 7)

# creating non encoded images appropriate for clusters
non_encoded_images = X_test.reshape(len(X_test), -1)
# convert each image to 1 dimensional array
non_encoded_images = non_encoded_images.astype(float) / 255.

# writing results to .xlsx file
for name, algorithm in clustering_algorithms:
    clustered_data = clustering(algorithm, encoded_images)
    sc, ch, db, hmg = perfomance_score(encoded_images, clustered_data, y_test)
    write_excel(name, 'Encoded', sc, ch, db, hmg)
    plot_confusion_matrix_clusters_images(y_test, clustered_data)

for name, algorithm in clustering_algorithms:
    clustered_data = clustering(algorithm, non_encoded_images)
    sc, ch, db, hmg = perfomance_score(non_encoded_images, clustered_data, y_test)
    write_excel(name, 'Non-Encoded', sc, ch, db, hmg)
    plot_confusion_matrix_clusters_images(y_test, clustered_data)

workbook.close()
