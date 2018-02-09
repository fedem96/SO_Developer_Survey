import numpy as np
import pickle

from utilities import readCSV, filterColumns, matrixFromStringsToNumbers


def prepare(fileDataset, separator, nameColSalary, nonAvailableValue, acceptedNA, fileX, fileY):
    ''' reading dataset from the csv file '''
    x, y = readCSV(fileDataset, separator, nameColSalary)
    # the dataset x is a list of rows, each row is a list of strings
    # y is the list of salaries
    # xLabels is the list of the names of the columns

    ''' calculation of the median salary'''
    medianSalary = np.median(y)

    ''' removing columns with too many non available elements '''
    x = filterColumns(x, nonAvailableValue, acceptedNA)

    ''' transforming x from a matrix of string to a matrix of values'''
    x = matrixFromStringsToNumbers(x)

    ''' transforming y: y[i] becomes 1 if i-th salary is above the median salary, otherwise 0 '''
    for i in range(len(y)):
        if y[i] > medianSalary:
            y[i] = 1
        else:
            y[i] = 0

    ''' saving x and y on files '''
    pickle.dump(x, open(fileX, "wb"))
    pickle.dump(y, open(fileY, "wb"))