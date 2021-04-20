# /--------------------------------------/
# NAME: Antonis Agapiou
# DEPARTMENT: INFORMATICS AND COMPUTER ENGINEERING
# E-MAIL: cs141081@uniwa.gr
# A.M: 711141081
# CLASS: PRML
# /--------------------------------------/

# import necessary libraries
import pandas as pd
import keras
import xlsxwriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import matplotlib.pyplot as plt    #plotting tools
from keras.utils import plot_model
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from ann_visualizer.visualize import ann_viz
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



# now create train and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(inputData, outputData, random_state=60)
from sklearn.preprocessing import MinMaxScaler

counter_train=0

for i in y_train:
    if (i == 1):
        counter_train=counter_train+1


counter_test = 0

for i in y_test:
    if (i == 1):
        counter_test = counter_test + 1

from imblearn.under_sampling import RandomUnderSampler

sampling_strategy = {0:(counter_train*3),1:counter_train}
nr = RandomUnderSampler(random_state=40,sampling_strategy=sampling_strategy)

X_train, y_train = nr.fit_sample(X_train, y_train.ravel())
print("After Undersampling, counts of label '1': {}".format(sum(y_train == 1)))
print("After Undersampling, counts of label '0': {}".format(sum(y_train == 0)))
# pd.value_counts(y_train).plot(kind="bar")
# plt.show()

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

row=0
col=0
workbook = xlsxwriter.Workbook('Output_v2.xlsx')
worksheet = workbook.add_worksheet()

worksheet.write(row, col,     'Classifier name')
worksheet.write(row, col + 1, 'Training or test set')
worksheet.write(row, col + 2, 'Number of training samples')
worksheet.write(row, col + 3, 'Number of non-healthy companies in training sample')
worksheet.write(row, col + 4, 'TP')
worksheet.write(row, col + 5, 'TN')
worksheet.write(row, col + 6, 'FP')
worksheet.write(row, col + 7, 'FN')
worksheet.write(row, col + 8, 'Precision')
worksheet.write(row, col + 9, 'Recall')
worksheet.write(row, col + 10, 'F1 Score')
worksheet.write(row, col + 11, 'Accuracy')

row = row + 1

from sklearn.linear_model import  LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = logreg.predict(X_train)
y_pred_test = logreg.predict(X_test)
# calculate the scores
tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_pred_train).ravel()
tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_pred_test).ravel()
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
pre_train = precision_score(y_train, y_pred_train, average='macro')
pre_test = precision_score(y_test, y_pred_test, average='macro')
rec_train = recall_score(y_train, y_pred_train, average='macro')
rec_test = recall_score(y_test, y_pred_test, average='macro')
f1_train = f1_score(y_train, y_pred_train, average='macro')
f1_test = f1_score(y_test, y_pred_test, average='macro')

#print the scores to the excel file

#first for the train set
worksheet.write(row, col,     'Logistic Regression')
worksheet.write(row, col + 1, 'Train Set')
worksheet.write(row, col + 2, len(X_train))
worksheet.write(row, col + 3, counter_train)
worksheet.write(row, col + 4, tp_train)
worksheet.write(row, col + 5, tn_train)
worksheet.write(row, col + 6, fp_train)
worksheet.write(row, col + 7, fn_train)
worksheet.write(row, col + 8, '{:.2f}'.format(pre_train))
worksheet.write(row, col + 9, '{:.2f}'.format(rec_train))
worksheet.write(row, col + 10, '{:.2f}'.format(f1_train))
worksheet.write(row, col + 11, '{:.2f}'.format(acc_train))
row=row+1
#and then for the test set
worksheet.write(row, col,     'Logistic Regression')
worksheet.write(row, col + 1, 'Test Set')
worksheet.write(row, col + 2, len(X_test))
worksheet.write(row, col + 3, counter_test)
worksheet.write(row, col + 4, tp_test)
worksheet.write(row, col + 5, tn_test)
worksheet.write(row, col + 6, fp_test)
worksheet.write(row, col + 7, fn_test)
worksheet.write(row, col + 8, '{:.2f}'.format(pre_test))
worksheet.write(row, col + 9, '{:.2f}'.format(rec_test))
worksheet.write(row, col + 10, '{:.2f}'.format(f1_test))
worksheet.write(row, col + 11, '{:.2f}'.format(acc_test))
row=row+1



from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
# calculate the scores
tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_pred_train).ravel()
tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_pred_test).ravel()
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
pre_train = precision_score(y_train, y_pred_train, average='macro')
pre_test = precision_score(y_test, y_pred_test, average='macro')
rec_train = recall_score(y_train, y_pred_train, average='macro')
rec_test = recall_score(y_test, y_pred_test, average='macro')
f1_train = f1_score(y_train, y_pred_train, average='macro')
f1_test = f1_score(y_test, y_pred_test, average='macro')

#print the scores to the excel file

#first for the train set
worksheet.write(row, col,     'Decision Tree Classifier')
worksheet.write(row, col + 1, 'Train Set')
worksheet.write(row, col + 2, len(X_train))
worksheet.write(row, col + 3, counter_train)
worksheet.write(row, col + 4, tp_train)
worksheet.write(row, col + 5, tn_train)
worksheet.write(row, col + 6, fp_train)
worksheet.write(row, col + 7, fn_train)
worksheet.write(row, col + 8, '{:.2f}'.format(pre_train))
worksheet.write(row, col + 9, '{:.2f}'.format(rec_train))
worksheet.write(row, col + 10, '{:.2f}'.format(f1_train))
worksheet.write(row, col + 11, '{:.2f}'.format(acc_train))
row=row+1
#and then for the test set
worksheet.write(row, col,     'Decision Tree Classifier')
worksheet.write(row, col + 1, 'Test Set')
worksheet.write(row, col + 2, len(X_test))
worksheet.write(row, col + 3, counter_test)
worksheet.write(row, col + 4, tp_test)
worksheet.write(row, col + 5, tn_test)
worksheet.write(row, col + 6, fp_test)
worksheet.write(row, col + 7, fn_test)
worksheet.write(row, col + 8, '{:.2f}'.format(pre_test))
worksheet.write(row, col + 9, '{:.2f}'.format(rec_test))
worksheet.write(row, col + 10, '{:.2f}'.format(f1_test))
worksheet.write(row, col + 11, '{:.2f}'.format(acc_test))
row=row+1

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)
# calculate the scores
tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_pred_train).ravel()
tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_pred_test).ravel()
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
pre_train = precision_score(y_train, y_pred_train, average='macro')
pre_test = precision_score(y_test, y_pred_test, average='macro')
rec_train = recall_score(y_train, y_pred_train, average='macro')
rec_test = recall_score(y_test, y_pred_test, average='macro')
f1_train = f1_score(y_train, y_pred_train, average='macro')
f1_test = f1_score(y_test, y_pred_test, average='macro')

#print the scores to the excel file

#first for the train set
worksheet.write(row, col,     'KNeighborsClassifier')
worksheet.write(row, col + 1, 'Train Set')
worksheet.write(row, col + 2, len(X_train))
worksheet.write(row, col + 3, counter_train)
worksheet.write(row, col + 4, tp_train)
worksheet.write(row, col + 5, tn_train)
worksheet.write(row, col + 6, fp_train)
worksheet.write(row, col + 7, fn_train)
worksheet.write(row, col + 8, '{:.2f}'.format(pre_train))
worksheet.write(row, col + 9, '{:.2f}'.format(rec_train))
worksheet.write(row, col + 10, '{:.2f}'.format(f1_train))
worksheet.write(row, col + 11, '{:.2f}'.format(acc_train))
row=row+1
#and then for the test set
worksheet.write(row, col,     'KNeighborsClassifier')
worksheet.write(row, col + 1, 'Test Set')
worksheet.write(row, col + 2, len(X_test))
worksheet.write(row, col + 3, counter_test)
worksheet.write(row, col + 4, tp_test)
worksheet.write(row, col + 5, tn_test)
worksheet.write(row, col + 6, fp_test)
worksheet.write(row, col + 7, fn_test)
worksheet.write(row, col + 8, '{:.2f}'.format(pre_test))
worksheet.write(row, col + 9, '{:.2f}'.format(rec_test))
worksheet.write(row, col + 10, '{:.2f}'.format(f1_test))
worksheet.write(row, col + 11, '{:.2f}'.format(acc_test))
row=row+1


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = lda.predict(X_train)
y_pred_test = lda.predict(X_test)
# calculate the scores
tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_pred_train).ravel()
tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_pred_test).ravel()
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
pre_train = precision_score(y_train, y_pred_train, average='macro')
pre_test = precision_score(y_test, y_pred_test, average='macro')
rec_train = recall_score(y_train, y_pred_train, average='macro')
rec_test = recall_score(y_test, y_pred_test, average='macro')
f1_train = f1_score(y_train, y_pred_train, average='macro')
f1_test = f1_score(y_test, y_pred_test, average='macro')

#print the scores to the excel file

#first for the train set
worksheet.write(row, col,     'LinearDiscriminantAnalysis')
worksheet.write(row, col + 1, 'Train Set')
worksheet.write(row, col + 2, len(X_train))
worksheet.write(row, col + 3, counter_train)
worksheet.write(row, col + 4, tp_train)
worksheet.write(row, col + 5, tn_train)
worksheet.write(row, col + 6, fp_train)
worksheet.write(row, col + 7, fn_train)
worksheet.write(row, col + 8, '{:.2f}'.format(pre_train))
worksheet.write(row, col + 9, '{:.2f}'.format(rec_train))
worksheet.write(row, col + 10, '{:.2f}'.format(f1_train))
worksheet.write(row, col + 11, '{:.2f}'.format(acc_train))
row=row+1
#and then for the test set
worksheet.write(row, col,     'LinearDiscriminantAnalysis')
worksheet.write(row, col + 1, 'Test Set')
worksheet.write(row, col + 2, len(X_test))
worksheet.write(row, col + 3, counter_test)
worksheet.write(row, col + 4, tp_test)
worksheet.write(row, col + 5, tn_test)
worksheet.write(row, col + 6, fp_test)
worksheet.write(row, col + 7, fn_test)
worksheet.write(row, col + 8, '{:.2f}'.format(pre_test))
worksheet.write(row, col + 9, '{:.2f}'.format(rec_test))
worksheet.write(row, col + 10, '{:.2f}'.format(f1_test))
worksheet.write(row, col + 11, '{:.2f}'.format(acc_test))
row=row+1

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = gnb.predict(X_train)
y_pred_test = gnb.predict(X_test)
# calculate the scores
tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_pred_train).ravel()
tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_pred_test).ravel()
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
pre_train = precision_score(y_train, y_pred_train, average='macro')
pre_test = precision_score(y_test, y_pred_test, average='macro')
rec_train = recall_score(y_train, y_pred_train, average='macro')
rec_test = recall_score(y_test, y_pred_test, average='macro')
f1_train = f1_score(y_train, y_pred_train, average='macro')
f1_test = f1_score(y_test, y_pred_test, average='macro')

#print the scores to the excel file

#first for the train set
worksheet.write(row, col,     'GaussianNB')
worksheet.write(row, col + 1, 'Train Set')
worksheet.write(row, col + 2, len(X_train))
worksheet.write(row, col + 3, counter_train)
worksheet.write(row, col + 4, tp_train)
worksheet.write(row, col + 5, tn_train)
worksheet.write(row, col + 6, fp_train)
worksheet.write(row, col + 7, fn_train)
worksheet.write(row, col + 8, '{:.2f}'.format(pre_train))
worksheet.write(row, col + 9, '{:.2f}'.format(rec_train))
worksheet.write(row, col + 10, '{:.2f}'.format(f1_train))
worksheet.write(row, col + 11, '{:.2f}'.format(acc_train))
row=row+1
#and then for the test set
worksheet.write(row, col,     'GaussianNB')
worksheet.write(row, col + 1, 'Test Set')
worksheet.write(row, col + 2, len(X_test))
worksheet.write(row, col + 3, counter_test)
worksheet.write(row, col + 4, tp_test)
worksheet.write(row, col + 5, tn_test)
worksheet.write(row, col + 6, fp_test)
worksheet.write(row, col + 7, fn_test)
worksheet.write(row, col + 8, '{:.2f}'.format(pre_test))
worksheet.write(row, col + 9, '{:.2f}'.format(rec_test))
worksheet.write(row, col + 10, '{:.2f}'.format(f1_test))
worksheet.write(row, col + 11, '{:.2f}'.format(acc_test))
row=row+1

from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, y_train)  # fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = gnb.predict(X_train)
y_pred_test = gnb.predict(X_test)
# calculate the scores
tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_pred_train).ravel()
tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_pred_test).ravel()
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
pre_train = precision_score(y_train, y_pred_train, average='macro')
pre_test = precision_score(y_test, y_pred_test, average='macro')
rec_train = recall_score(y_train, y_pred_train, average='macro')
rec_test = recall_score(y_test, y_pred_test, average='macro')
f1_train = f1_score(y_train, y_pred_train, average='macro')
f1_test = f1_score(y_test, y_pred_test, average='macro')

#print the scores to the excel file

#first for the train set
worksheet.write(row, col,     'SVM')
worksheet.write(row, col + 1, 'Train Set')
worksheet.write(row, col + 2, len(X_train))
worksheet.write(row, col + 3, counter_train)
worksheet.write(row, col + 4, tp_train)
worksheet.write(row, col + 5, tn_train)
worksheet.write(row, col + 6, fp_train)
worksheet.write(row, col + 7, fn_train)
worksheet.write(row, col + 8, '{:.2f}'.format(pre_train))
worksheet.write(row, col + 9, '{:.2f}'.format(rec_train))
worksheet.write(row, col + 10, '{:.2f}'.format(f1_train))
worksheet.write(row, col + 11, '{:.2f}'.format(acc_train))
row=row+1
#and then for the test set
worksheet.write(row, col,     'SVM')
worksheet.write(row, col + 1, 'Test Set')
worksheet.write(row, col + 2, len(X_test))
worksheet.write(row, col + 3, counter_test)
worksheet.write(row, col + 4, tp_test)
worksheet.write(row, col + 5, tn_test)
worksheet.write(row, col + 6, fp_test)
worksheet.write(row, col + 7, fn_test)
worksheet.write(row, col + 8, '{:.2f}'.format(pre_test))
worksheet.write(row, col + 9, '{:.2f}'.format(rec_test))
worksheet.write(row, col + 10, '{:.2f}'.format(f1_test))
worksheet.write(row, col + 11, '{:.2f}'.format(acc_test))
row=row+1
CustomModel = keras.models.Sequential()
CustomModel.add(keras.layers.Dense(16, input_dim=X_train.shape[1], activation='relu'))
CustomModel.add(keras.layers.Dense(2, activation='softmax'))
# display the architecture
# print(CustomModel.summary())
# compile model using accuracy to measure model performance
CustomModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
#keras.utils.np_utils.to_catergorical(y_train) coverts a class vector to binary class matrix(for categorical_crossentropy)

CustomModel.fit(X_train, keras.utils.np_utils.to_categorical(y_train), epochs=100, verbose=False)

y_pred_train = CustomModel.predict_classes(X_train)
y_pred_test = CustomModel.predict_classes(X_test)
# now check for both train and test data, how well the model learned the patterns
ann_viz(CustomModel)
tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_pred_train).ravel()
tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_pred_test).ravel()
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
pre_train = precision_score(y_train, y_pred_train, average='macro')
pre_test = precision_score(y_test, y_pred_test, average='macro')
rec_train = recall_score(y_train, y_pred_train, average='macro')
rec_test = recall_score(y_test, y_pred_test, average='macro')
f1_train = f1_score(y_train, y_pred_train, average='macro')
f1_test = f1_score(y_test, y_pred_test, average='macro')

#print the scores to the excel file

#first for the train set
worksheet.write(row, col,     'ANN classifier')
worksheet.write(row, col + 1, 'Train Set')
worksheet.write(row, col + 2, len(X_train))
worksheet.write(row, col + 3, counter_train)
worksheet.write(row, col + 4, tp_train)
worksheet.write(row, col + 5, tn_train)
worksheet.write(row, col + 6, fp_train)
worksheet.write(row, col + 7, fn_train)
worksheet.write(row, col + 8, '{:.2f}'.format(pre_train))
worksheet.write(row, col + 9, '{:.2f}'.format(rec_train))
worksheet.write(row, col + 10, '{:.2f}'.format(f1_train))
worksheet.write(row, col + 11, '{:.2f}'.format(acc_train))
row=row+1
#and then for the test set
worksheet.write(row, col,     'ANN classifier')
worksheet.write(row, col + 1, 'Test Set')
worksheet.write(row, col + 2, len(X_test))
worksheet.write(row, col + 3, counter_test)
worksheet.write(row, col + 4, tp_test)
worksheet.write(row, col + 5, tn_test)
worksheet.write(row, col + 6, fp_test)
worksheet.write(row, col + 7, fn_test)
worksheet.write(row, col + 8, '{:.2f}'.format(pre_test))
worksheet.write(row, col + 9, '{:.2f}'.format(rec_test))
worksheet.write(row, col + 10, '{:.2f}'.format(f1_test))
worksheet.write(row, col + 11, '{:.2f}'.format(acc_test))

workbook.close()