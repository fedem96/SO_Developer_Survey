import csv
from timeit import default_timer as timer

import numpy as np
from sklearn import preprocessing


# reads the csv file containing the dataset
# returns the dataset in a list of rows (without salary column), and the salary column
def readCSV(fileDataset, delimiter, nameColSalary):
    start = timer()

    x = []
    y = []
    with open(fileDataset) as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)

        xLabels = next(reader)
        salaryIndex = xLabels.index(nameColSalary)
        xLabels.pop(salaryIndex)
        for row in reader:
            try:
                salary = float(row.pop(salaryIndex))
                x.append(row)
                y.append(salary)
            except ValueError:
                None

    end = timer()
    print("csv reading time: ", (end-start))

    return x, y


# removes columns from the dataset with too much non available values
# returns the dataset filtered
def filterColumns(x, nonAvailableValue, acceptedNA):
    start = timer()
    xTransp = np.transpose(x)
    xReturn = []

    for col in xTransp:
        if not containsTooMuchNA(col, nonAvailableValue, acceptedNA):
            xReturn.append(col)

    xReturn = np.transpose(xReturn)

    end = timer()
    print("filtering columns time: ", (end - start))

    return xReturn


# given a column, value of non available values, and maximum accepted non available,
# returns True if column has enough valid values, otherwise False
def containsTooMuchNA(column, nonAvailableValue, acceptedNA):
    numNA = 0
    for cell in column:
        if cell == nonAvailableValue:
            numNA += 1
    return (numNA / len(column)) > acceptedNA


# given the dataset with string values, transforms it into a dataset with integer values
# returns the dataset transformed
def matrixFromStringsToNumbers(x):

    ''' label encoders creation '''
    start = timer()
    les = []  # Label EncoderS

    numCols = len(x[0])
    for c in range(numCols):
        # le is a LabelEncoder: it allows to transform in numbers the strings of i-th column, and vice versa
        le = preprocessing.LabelEncoder()
        le.fit([x[r][c] for r in range(len(x))])
        les.append(le)
    # after the for loop, in les there is a LabelEncoder for each dataset column

    end = timer()
    print("LabelEncoders creation time: ", (end - start))

    ''' effective x transformation '''
    start = timer()
    # each column of x is transformed and put in xTransp as a row
    xTransp = []
    for c in range(numCols):
        xTransp.append(les[c].transform([row[c] for row in x]))

    # xTransp contains numbers, but it's transposed compared to x, so the numbered x is obtained by transposing xTransp
    x = list(np.array(xTransp).transpose())

    end = timer()
    print("x transformation time: ", (end - start))

    return x


# splits a list into two separated lists
# the first list contains all the elements from the beginning to beginSplit-1, and from endSplit to the end
# the second list contains all the elements non included in the first list
# returns the first list and the second list
def split(x, beginSplit=0, endSplit=None):
    if endSplit is None:
        endSplit = len(x)

    x1 = x[:beginSplit]
    x1.extend(x[endSplit:])
    x2 = x[beginSplit:endSplit]

    return x1, x2


# tests accuracy of a classifier
# x: values to be predicted
# y: desired values
# returns the accuracy, between 0 and 1
def testClassifierAccuracy(classifier, x, y):
    
    ''' training dataset prediction with decision tree '''
    prediction = classifier.predict(x)

    ''' prediction and expected values comparison '''
    return calculateAccuracy(y, prediction)


# calculates fraction of equal elements in same positions in two lists
# returns a number between 0 and 1
def calculateAccuracy(y, z):
    equals = 0
    l = len(y)
    if l != len(z):
        raise Exception("input lists must have the same length!")
    for i in range(l):
        if y[i] == z[i]:
            equals += 1
    return equals/l
