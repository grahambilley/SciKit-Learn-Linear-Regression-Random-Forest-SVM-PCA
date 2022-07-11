## Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms to detect eye state

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

######################################### Reading and Splitting the Data ###############################################
# XXX
# TODO: Read in all the data. Replace the 'xxx' with the path to the data set.
# XXX
data = pd.read_csv('eeg_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data.
random_state = 100

# XXX
# TODO: Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 100.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=random_state)
# XXX


# ############################################### Linear Regression ###################################################
# XXX
# TODO: Create a LinearRegression classifier and train it.
reg = LinearRegression().fit(x_train, y_train)
# XXX


# XXX
# TODO: Test its accuracy (on the training set) using the accuracy_score method.
# TODO: Test its accuracy (on the testing set) using the accuracy_score method.
# Note: Round the output values greater than or equal to 0.5 to 1 and those less than 0.5 to 0. You can use y_predict.round() or any other method.
reg_train_preds = reg.predict(x_train).round()
reg_train_acc = accuracy_score(y_train, reg_train_preds) # 0.64 Q3.1
reg_test_preds  = reg.predict(x_test).round()
reg_test_acc = accuracy_score(y_test, reg_test_preds) # 0.64 Q3.1
# XXX


# ############################################### Random Forest Classifier ##############################################
# XXX
# TODO: Create a RandomForestClassifier and train it.
rf = RandomForestClassifier().fit(x_train, y_train)
# XXX


# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
rf_train_preds = rf.predict(x_train)
rf_train_acc = accuracy_score(y_train, rf_train_preds) # 1.00 Q3.1
rf_test_preds  = rf.predict(x_test)
rf_test_acc = accuracy_score(y_test, rf_test_preds) # 0.90 Q3.1
# print('train: ', rf_train_acc)
# print('test: ', rf_test_acc)
# XXX


# XXX
# TODO: Determine the feature importance as evaluated by the Random Forest Classifier.
#       Sort them in the descending order and print the feature numbers. The report the most important and the least important feature.
#       Mention the features with the exact names, e.g. X11, X1, etc.
#       Hint: There is a direct function available in sklearn to achieve this. Also checkout argsort() function in Python.
imp = rf.feature_importances_
# print(imp,'\n')
asort = np.argsort(-imp) # '-' for descending order
# print(asort,'\n')
# print(x_train.columns)
print(x_train.columns[asort]) # ['X7', 'X6', 'X2', 'X13', 'X14', 'X1', 'X4', 'X12', 'X11', 'X8', 'X5', 'X3', 'X10', 'X9']
# XXX


# XXX
# TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth'.
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
rf_parameters = {'n_estimators':[1,2,3,4,5,6,7,8,9,10,11,12,13,14], 'max_depth':[20,30,50]}
rf_clf = GridSearchCV(RandomForestClassifier(), rf_parameters , cv=10)
rf_clf_tuned = rf_clf.fit(x_train, y_train)
print(rf_clf_tuned.best_params_) # {'max_depth': 50, 'n_estimators': 13}
print(rf_clf_tuned.best_score_)  # 0.90

rf2 = RandomForestClassifier(n_estimators = 13, max_depth=50).fit(x_train, y_train)
rf2_train_preds = rf2.predict(x_train)
rf2_train_acc = accuracy_score(y_train, rf2_train_preds) # 1.00 Q3.2
rf2_test_preds  = rf2.predict(x_test)
rf2_test_acc = accuracy_score(y_test, rf2_test_preds) # 0.91 Q3.2
# print('train2: ', rf2_train_acc)
# print('test2: ', rf2_test_acc)
# XXX


# ############################################ Support Vector Machine ###################################################
# XXX
# TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
# TODO: Create a SVC classifier and train it.
scaler = StandardScaler()
scaler.fit(x_train)
svm = SVC().fit(scaler.transform(x_train), y_train)
# XXX


# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
svm_train_preds = svm.predict(scaler.transform(x_train))
svm_train_acc = accuracy_score(y_train, svm_train_preds) # 0.71 Q3.1
svm_test_preds  = svm.predict(scaler.transform(x_test))
svm_test_acc = accuracy_score(y_test, svm_test_preds) # 0.71 Q3.1
# print('train: ', svm_train_acc)
# print('test: ', svm_test_acc)
# XXX


# XXX
# TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
svm_parameters = {'kernel':('linear', 'rbf'), 'C':[0.01, 1, 100]}
svm_clf = GridSearchCV(SVC(), svm_parameters , cv=10, n_jobs = -1)
svm_clf_tuned = svm_clf.fit(scaler.transform(x_train), y_train)
print(svm_clf_tuned.best_params_) # {'C': 100, 'kernel': 'rbf'}
print(svm_clf_tuned.best_score_) # 0.82

svm2 = SVC(kernel = 'rbf', C = 100).fit(scaler.transform(x_train), y_train)
svm_train_preds2 = svm2.predict(scaler.transform(x_train))
svm_train_acc2 = accuracy_score(y_train, svm_train_preds2) # 0.83 Q3.2
svm_test_preds2  = svm2.predict(scaler.transform(x_test))
svm_test_acc2 = accuracy_score(y_test, svm_test_preds2) # 0.83 Q3.2
# print('train: ', svm_train_acc2)
# print('test: ', svm_test_acc2)

# print(svm_clf.cv_results_['params']) # index 5 has combination of best parameters
# print("mean training score: ",svm_clf.cv_results_['mean_train_score'][5]) # 0.83 Q3.3
# print("mean testing score: ",svm_clf.cv_results_['mean_test_score'][5]) # 0.82 Q3.3
# print("mean fit time: ",svm_clf.cv_results_['mean_fit_time'][5]) # 7.80 Q3.3
# XXX


# ######################################### Principal Component Analysis #################################################
# XXX
# TODO: Perform dimensionality reduction of the data using PCA.
#       Set parameters n_component to 10 and svd_solver to 'full'. Keep other parameters at their default value.
#       Print the following arrays:
#       - Percentage of variance explained by each of the selected components
#       - The singular values corresponding to each of the selected components.
pca = PCA(n_components = 10, svd_solver = 'full').fit(x_data)
print(np.round(pca.explained_variance_ratio_,2),'\n') # [0.51, 0.38, 0.12, 0., 0., 0., 0., 0., 0., 0.] 
print(pca.singular_values_) # [886690.55021511, 765870.22149031, 428019.7135883, 8460.03827621, 5477.2458465, 4180.81523164, 3249.68937137, 1714.82156063, 1548.48148676, 1132.55981354]
# XXX


