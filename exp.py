from random import randint

import pickle

from os.path import isfile
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from matplotlib import pyplot as plt

from prepare import prepare
from utilities import *

start = timer()

''' preparation parameters '''
fileDataset = "dataset/survey_results_public.csv"       # path of csv file that contains the dataset
separator = ","                                         # separator character in the csv file
nameColSalary = "Salary"                                # name of the column of the salary
nonAvailableValue = "NA"                                # string that represents non available values in the csv file
acceptedNA = 0.05           # fraction of maximum accepted Non Available values for each column of the dataset: if not satisfied, column is discarded

''' preparation and experiment parameters '''
fileX = "x.p"
fileY = "y.p"

''' experiment parameters '''
K = 10                      # parameter K for K-Fold cross validation
numTests = 2               # number of tests to be done, the final results are the average obtained in the test
maxDepth = 6                # maximum depth for every tree, useful to avoid overfitting
minSamplesLeaf = 61         # minimum samples for every leaf, useful to avoid overfitting
nEstimators = 20            # number of estimators (trees) in every random forest

''' possible creation of intermediate files, only needed if at least one of them does not exist '''
if not(isfile(fileX) and isfile(fileY)):
    prepare(fileDataset, separator, nameColSalary, nonAvailableValue, acceptedNA, fileX, fileY)

''' reading of intermediate files '''
x = pickle.load(open(fileX, "rb"))
y = pickle.load(open(fileY, "rb"))

print("effective number of used columns: ", len(x[0]))

datasetLength = len(x)

avgTrainingSetAccuracyDT = 0
avgTestSetAccuracyDT = 0

avgTrainingSetAccuracyRF = 0
avgTestSetAccuracyRF = 0

print("will be executed", numTests, "tests")
for j in range(numTests):

    print("running test", j)

    ''' mixing the dataset randomly, keeping the matches with y '''
    for i in range(datasetLength):
        randomIndex = randint(i, datasetLength - 1)
        x[i], x[randomIndex] = x[randomIndex], x[i]
        y[i], y[randomIndex] = y[randomIndex], y[i]

    ''' calculating decision tree accuracies with K-fold cross-validation '''
    dt = DecisionTreeClassifier(max_depth=maxDepth, min_samples_leaf=minSamplesLeaf)
    kFoldCrossValidationDT = cross_validate(dt, x, y, cv=K, return_train_score=True)
    avgTrainingSetAccuracyDT += np.mean(kFoldCrossValidationDT['train_score'])
    avgTestSetAccuracyDT += np.mean(kFoldCrossValidationDT['test_score'])

    ''' calculating random forest accuracies with K-fold cross-validation '''
    rf = RandomForestClassifier(n_estimators=nEstimators, max_depth=maxDepth, min_samples_leaf=minSamplesLeaf)
    kFoldCrossValidationRF = cross_validate(rf, x, y, cv=K, return_train_score=True)
    avgTrainingSetAccuracyRF += np.mean(kFoldCrossValidationRF['train_score'])
    avgTestSetAccuracyRF += np.mean(kFoldCrossValidationRF['test_score'])

avgTrainingSetAccuracyDT /= numTests
avgTestSetAccuracyDT /= numTests

avgTrainingSetAccuracyRF /= numTests
avgTestSetAccuracyRF /= numTests

end = timer()
print("total time: ", (end-start))


''' experiment output '''
print("average train dataset prediction accuracy with decision tree: ", avgTrainingSetAccuracyDT)
print("average test dataset prediction accuracy with decision tree: ", avgTestSetAccuracyDT)

print("average train dataset prediction accuracy with random forest: ", avgTrainingSetAccuracyRF)
print("average test dataset prediction accuracy with random forest: ", avgTestSetAccuracyRF)

averageAccuracies = [avgTrainingSetAccuracyDT, avgTestSetAccuracyDT, avgTrainingSetAccuracyRF, avgTestSetAccuracyRF]

''' plotting the histogram '''
plt.bar([0, 1, 3, 4], averageAccuracies,
        tick_label=["Train", "Test", "Train", "Test"])
plt.title("Prediction accuracy")
plt.xlabel("Decision Tree Sets ←--   --→ Random Forest Sets")
plt.ylabel("Accuracy")
plt.show()
