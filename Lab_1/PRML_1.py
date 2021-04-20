
# /-------------------------/

# Name: Agapiou Antonis

# Department: Informatics and Computer Engineering

# E-mail: cs141081@uniwa.gr

# A.M: 711141081

# /-------------------------/


# Importing necessary python packages

import math

from sklearn.metrics import mean_absolute_error as mae

from sklearn.metrics import mean_squared_error as mse

import numpy as np

import scipy.optimize as opt

import matplotlib.pyplot as plt


# +++++++++++++ start of custom functions +++++++++


# first function: -(x-3)^2 + 2

def custFunc1(x):
    return np.negative(np.power((x - 3), 2)) + 2


# second function: sin(x) + 1/e^x

def custFunc2(x):
    return np.sin(x) + (1 / np.exp(x))


# 3 degree polynomial: y = ax^3 + bx^2 +cx + d

def poly3degree(x, a, b, c, d):
    return a * np.power(x, 3) + b * np.power(x, 2) + c * x + d


# 4th degree polynomial: y = ax^4 + bx^3 + cx^2 + dx + e

def poly4degree(x, a, b, c, d, e):
    return a * np.power(x, 4) + b * np.power(x, 3) + c * np.power(x, 2) + d * x + e


# +++++++++++++ end of custom functions +++++++++


# generate 100 random values from 0 to 33 using uniform distribution

inputValues_unsorted = np.random.uniform(0., 33., 100)

# sorting the values in ascending order using the numpy.sort() function

inputValues = np.sort(inputValues_unsorted, axis=None)

# generate outcome values using our functions and inputValues

Func1Output = custFunc1(inputValues)

Func2Output = custFunc2(inputValues)

# create the functions' output plots

plt.plot(inputValues, Func1Output, '*', label='Func1\'s output')

plt.plot(inputValues, Func2Output, '+', label='Func2\'s output')

plt.legend(loc='best')

plt.title('Functions\' outputs')

plt.xlabel('Input Values')

plt.ylabel('Output Values')

plt.show()
plt.waitforbuttonpress(2)

plt.close()

# save the results to a txt file

fileNameToSaveAndLoad = 'OutputResults.txt'

headerValues = 'Inputs Func1Output Func2Output'

valuesToSave = np.column_stack([inputValues, Func1Output, Func2Output])

# formatting the file using the fmt = "%8.2f" to align columns with header and to have

# only 2 decimal points

np.savetxt(fileNameToSaveAndLoad, valuesToSave, fmt="%8.2f", header=headerValues)

# load the output results from the OutputResults.txt file

availableValues = np.loadtxt(fileNameToSaveAndLoad)

# using scypi.optimization to fit models parameters

# first using the funtion1's output

best_vals_poly31, covar_poly31 = opt.curve_fit(poly3degree, availableValues[:, 0], availableValues[:, 1])

best_vals_poly41, covar_poly41 = opt.curve_fit(poly4degree, availableValues[:, 0], availableValues[:, 1])

# second using the function2's output

best_vals_poly32, covar_poly32 = opt.curve_fit(poly3degree, availableValues[:, 0], availableValues[:, 2])

best_vals_poly42, covar_poly42 = opt.curve_fit(poly4degree, availableValues[:, 0], availableValues[:, 2])

# Estimate the outcomes using the fitted models

# first using the custfunc1's output

outputValuesPredicted_p3_f1 = poly3degree(availableValues[:, 0], best_vals_poly31[0], best_vals_poly31[1],
                                          best_vals_poly31[2],

                                          best_vals_poly31[3])

outputValuesPredicted_p4_f1 = poly4degree(availableValues[:, 0], best_vals_poly41[0], best_vals_poly41[1],
                                          best_vals_poly41[2],

                                          best_vals_poly41[3], best_vals_poly41[4])

outputValuesPredicted_p3_f2 = poly3degree(availableValues[:, 0], best_vals_poly32[0], best_vals_poly32[1],
                                          best_vals_poly32[2],

                                          best_vals_poly32[3])

outputValuesPredicted_p4_f2 = poly4degree(availableValues[:, 0], best_vals_poly42[0], best_vals_poly42[1],
                                          best_vals_poly42[2],

                                          best_vals_poly42[3], best_vals_poly42[4])

# plotting

plt.plot(availableValues[:, 0], availableValues[:, 1], label='Func1\'s Output')

plt.plot(availableValues[:, 0], outputValuesPredicted_p3_f1, '*', label='polynomial 3rd degree')

plt.plot(availableValues[:, 0], outputValuesPredicted_p4_f1, '+', label='polynomial 4th degree')

plt.legend(loc='best')

plt.title('Models from polynomials using the outcomes from custFunc1 ')

plt.xlabel('Input Values')

plt.ylabel('Output Values')

plt.show()

plt.waitforbuttonpress(2)

plt.close()

plt.plot(availableValues[:, 0], availableValues[:, 2], label='Func2\'s Output')

plt.plot(availableValues[:, 0], outputValuesPredicted_p3_f2, '*', label='polynomial 3rd degree')

plt.plot(availableValues[:, 0], outputValuesPredicted_p4_f2, '+', label='polynomial 4th degree')

plt.legend(loc='best')

plt.title('Models from polynomials using the outcomes from custFunc2')

plt.xlabel('Input Values')

plt.ylabel('Output Values')

plt.show()

plt.waitforbuttonpress(2)

plt.close()

# mean absolute error

print('Mean absolute errors results: ')

print('Mean absolute error(custFunc1 - polynomial 3rd degree): ', mae(Func1Output, outputValuesPredicted_p3_f1))

print('Mean absolute error(custFunc1 - polynomial 4th degree): ', mae(Func1Output, outputValuesPredicted_p4_f1))

print('Mean absolute error(custFunc2 - polynomial 3rd degree): ', mae(Func2Output, outputValuesPredicted_p3_f2))

print('Mean absolute error(custFunc2 - polynomial 4th degree): ', mae(Func2Output, outputValuesPredicted_p4_f2))

# mean squared error

print('Mean root squared errors results: ')

print('Mean root squared error(custFunc1 - polynomial 3rd degree): ',math.sqrt(mse(Func1Output, outputValuesPredicted_p3_f1)))

print('Mean root squared error(custFunc1 - polynomial 4th degree): ',math.sqrt(mse(Func1Output, outputValuesPredicted_p4_f1)))

print('Mean root squared error(custFunc2 - polynomial 3rd degree): ',math.sqrt(mse(Func2Output, outputValuesPredicted_p3_f2)))

print('Mean root squared error(custFunc2 - polynomial 4th degree): ',math.sqrt(mse(Func2Output, outputValuesPredicted_p4_f2)))
