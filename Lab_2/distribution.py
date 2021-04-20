import numpy as np
import pandas as pd
# import keras
import xlsxwriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt    #plotting tools
import seaborn as sns

# read the data
fileName = 'Dataset2Use_Assignment2.xlsx'
sheetName = 'Total'
try:
    #confirming the file exists
    sheetValues = pd.read_excel(fileName, sheetName)
    print(' ... Successgul parsing of file:', fileName)
    print("Columns headings: ")
    print(sheetValues.columns)
except FileNotFoundError:
    print(FileNotFoundError)

inputData = sheetValues[sheetValues.columns[:-2]].values

# now convert the categorical values to unique class id and save the name-to-id match

outputData = sheetValues[sheetValues.columns[-2]]
outputData, levels = pd.factorize(outputData)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputData, outputData, random_state=0)


pd.value_counts(y_test).plot(kind="bar")
plt.show()


# describes info about train and test set
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

# apply near miss
from imblearn.under_sampling import RandomUnderSampler

sampling_strategy = {0:519,1:173}
nr = RandomUnderSampler(random_state=40,sampling_strategy=sampling_strategy)

X_train_miss, y_train_miss = nr.fit_sample(X_train, y_train.ravel())

print('After Undersampling, the shape of train_X: {}'.format(X_train_miss.shape))
print('After Undersampling, the shape of train_y: {} \n'.format(y_train_miss.shape))

print("After Undersampling, counts of label '1': {}".format(sum(y_train_miss == 1)))
print("After Undersampling, counts of label '0': {}".format(sum(y_train_miss == 0)))

pd.value_counts(y_train_miss).plot(kind="bar")
plt.show()



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


