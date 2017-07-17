#!/usr/bin/env python

#******************************************************************************
#        Lawrence Berkeley National Lab
#         Term:  VFP Student Summer 2017
#       Mentor:  Silvia Crivelli
# 
#    File Name:  protein_ML.py
# 
#  Programmers:  Bogdan Czejdo - Faculty
#                Casey Lorenzen - Student 2017
#                Catherine Spooner - Student 2017
#                Jim Inscoe - Student 2016
# 
#  For correspondence, contact Silvia Crivelli
# 
#  Revision     Date                        Release Comment
#  --------  ----------  ------------------------------------------------------
#    1.0     06/05/2016  Initial Release
#    2.0     06/13/2017  New Automation included
# 
#  File Description
#  ----------------
# 
# 
# ******************************************************************************

# Unconditionally Imported Packages

import pandas as pd
import numpy as np
from sklearn import preprocessing
import ProteinFunctions as pf
import sys
import itertools
import time
import os
import inspect
import argparse

# Constants - things that are given right now.  Possibly should be added to the parameters list

DROPOUT_ACTIVATION_RELU = 'relu'
DROPOUT_ACTIVATION_TANH = 'tanh'
LABEL_INDEX = 1

def createList(string):
    outList = []
    start_index = []
    end_index = []
    for i in range(0, len(string)):
        if (string[i] == "("):
            start_index.extend([i+1])
        if (string[i] == ")"):
            end_index.extend([i])
    if (len(start_index) != len(end_index)):
        print("Your pairs dont match")
    else:
        for j in range(0, len(start_index)):
            substring = string[start_index[j]:end_index[j]].split(',')
            substring = [int(i) for i in substring]
            start = substring[0]
            end = substring[1]+1
            outList.extend(range(start, end))
    return outList

def run_DropoutNN(X_train, X_test, Y_train, Y_test):
    import theano
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import SGD, Adam, RMSprop
    from keras.utils import np_utils

    # convert to numpy arrays    
    Y_train = np.array(Y_train)
    Y_test =  np.array(Y_test)
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # create a new sequential model 
    model = Sequential()

    # add a neural network layer of the size of our feature vector, 
    # using activation relu, because it is our first layer
    size=len(feature_cols)
    model.add(Dense(size, input_dim=size, activation = DROPOUT_ACTIVATION_RELU)) 
    model.add(Dropout(args.dropout_percent))
    
    # for this next layer, use half as many neurons as the first layer
    # also, use tanh for the activation function
    halfsize = size/2
    model.add(Dense(halfsize, activation = DROPOUT_ACTIVATION_TANH))
    model.add(Dropout(args.dropout_percent))

    # for this next layer, use a quarter as many neurons as the first layer
    # also, use relu for the activation function
    quartersize = size/4    
    model.add(Dense(quartersize, activation = DROPOUT_ACTIVATION_RELU))
    model.add(Dropout(args.dropout_percent))
    
    # for this final layer, use only one neuron
    model.add(Dense(args.dropout_final_layer))
    print 'Model initialized'

    # compiling means that we are done adding components to the model, 
    # and that the model will train using mse loss and an rmsprop optimizer
    model.compile(loss=args.dropout_loss, optimizer=args.dropout_optimizer) 

    # the fit function says that we will train the model and evaluate it 
    # with the following parameters, verbose lets us see where the model 
    # is in training
    hist = model.fit(X_train, Y_train, validation_data = (X_test,Y_test), 
                     batch_size=args.dropout_batch_size, epochs=args.dropout_epochs, verbose=1)
    print('Model trained')

    predictions = np.array(list(itertools.chain.from_iterable(model.predict(X_test))))

    return predictions

def run_SVR_ml(X_train, X_test, Y_train, Y_test):
    from sklearn.svm import SVR
    from sklearn.externals import joblib
    
    # Initialize SVM model with Radial Basis Kernel, epsilon = 0.1, and 
    # gamma of 0.1 (The best model in the paper by Shokufeh Mirzaeh and 
    # Silvia Crivelli)
    clf = SVR(kernel = args.svr_kernel, epsilon = args.svr_epsilon, gamma = args.svr_gamma)
    print 'Model initialized\n'

    # Train the model on the training feature vectors and their labels
    clf.fit(X_train, Y_train)
    print 'Model trained\n'

    # Save the model so you can invoke joblib.load(filename.pkl) 
    # later on
    #joblib.dump(clf, args.svr_savename)

    # predict labels of test feature vectors
    predictions = clf.predict(X_test)

    return predictions

def run_XGBoost_ml(X_train, X_test, Y_train, Y_test):
    import xgboost as xg
    from sklearn.externals import joblib

    # creates gradient boosting classifier (parameters should be grid
    # searched in the future to reduce loss)
    clf = xg.XGBRegressor(max_depth = args.xg_max_depth, learning_rate = args.xg_learning_rate, 
                          n_estimators = args.xg_estimators, nthread = args.xg_thread)
    print 'Model initialized'

    # train classifier on training set
    clf.fit(X_train, Y_train)
    print('Model trained')

    # Save the model so you can invoke joblib.load(filename.pkl) 
    # later on
    #joblib.dump(clf, args.xg_savename)
    
    predictions = clf.predict(X_test)
    
    return predictions 

parser = argparse.ArgumentParser(description="This program will check protein model correctness using machine learning techniques.")
parser.add_argument("filename", help='Database Filename', type=str)
parser.add_argument("MLmethod", help='Desired Machine Learning Method: SVR, Dropout, XGBOOST', type=str, choices=["SVR", "Dropout", "XGBOOST"])

parser.add_argument("feature_vectors", help='Column indices of all feature vectors')
parser.add_argument("training_rows", help='Training row indices')
parser.add_argument("testing_rows", help='Testing row indices')
parser.add_argument("proteinID", help='Column Index for protein ID', type=int)
parser.add_argument("--feature_col_remove", help='feature_col_remove -- Use this option if you wish to remove a column from the feature vector set.  Input the column index to remove', type=int, default=None)
parser.add_argument("--row_remove", help='row_remove -- Use this option if you wish to remove a row from the training rows.  Input the row index to remove', type=int, default=None)
parser.add_argument("--svr_kernel", help='svr_kernel -- This option should only be used if you are using SVR', default='rbf')
parser.add_argument("--svr_epsilon", help='svr_epsilon -- This option should only be used if you are using SVR', default=0.1, type=float)
parser.add_argument("--svr_gamma", help='svr_gamma -- This option should only be used if you are using SVR', default=0.1, type=float)
parser.add_argument("--svr_savename", help='svr_savename -- This option should only be used if you are using SVR', default='SVR.pkl')
parser.add_argument("--dropout_batch_size", help='dropout_batch_size -- This option should only be used if you are using Dropout', default=100, type=int)
parser.add_argument("--dropout_epochs", help='dropout_epochs -- This option should only be used if you are using Dropout', default=5, type=int)
parser.add_argument("--dropout_percent", help='dropout_percent -- This option should only be used if you are using Dropout', default=0.2, type=float)
parser.add_argument("--dropout_final_layer", help='dropout_final_layer -- This option should only be used if you are using Dropout', default=1, type=int)
parser.add_argument("--dropout_loss", help='dropout_loss -- This option should only be used if you are using Dropout', default='mse')
parser.add_argument("--dropout_optimizer", help='dropout_optimizer -- This option should only be used if you are using Dropout', default='rmsprop')
parser.add_argument("--xg_max_depth", help='xg_max_depth -- This option should only be used if you are using XGBOOST', default=6, type=int)
parser.add_argument("--xg_learning_rate", help='xg_learning_rate -- This option should only be used if you are using XGBOOST', default=0.1, type=float)
parser.add_argument("--xg_estimators", help='xg_estimators -- This option should only be used if you are using XGBOOST', default=10000, type=int)
parser.add_argument("--xg_thread", help='xg_thread -- This option should only be used if you are using XGBOOST', default=-1, type=int)
parser.add_argument("--xg_savename", help='xg_savename -- This option should only be used if you are using XGBOOST', default='XGBOOST.pkl')

args = parser.parse_args()

startTime=time.clock()

print('\nYou are running: ' + os.path.basename(inspect.getfile(inspect.currentframe()))) 
print('in the following path directory: ' + os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+ '\n')
print('Using ' + str(args.MLmethod) +' method \n')


AllData = pd.read_csv(args.filename)

feature_cols = createList(args.feature_vectors)
train_rows =  createList(args.training_rows)
test_rows = createList(args.testing_rows)

if (args.feature_col_remove is not None):
    feature_cols.remove(args.feature_col_remove)
if (args.row_remove is not None):
    train_rows.remove(args.row_remove)

X_train, X_test, Y_train, Y_test = AllData.iloc[train_rows, feature_cols], AllData.iloc[test_rows, feature_cols], AllData.iloc[train_rows, LABEL_INDEX],AllData.iloc[test_rows, LABEL_INDEX]

#print(args.proteinID)
proteinID = AllData.iloc[test_rows, args.proteinID]


# Create MinMax scaler
mnscaler = preprocessing.MinMaxScaler()
    
# Fit the minmax scaler onto the data
X_train = mnscaler.fit_transform(np.array(X_train))
X_test = mnscaler.fit_transform(np.array(X_test))
print('\nScaling accomplished \n')

# Run requested prediction

if args.MLmethod == "SVR":
    predictions = run_SVR_ml(X_train, X_test, Y_train, Y_test)

elif args.MLmethod == "Dropout":
    predictions = run_DropoutNN(X_train, X_test, Y_train, Y_test)

elif args.MLmethod == "XGBOOST":
    predictions = run_XGBoost_ml(X_train, X_test, Y_train, Y_test)


resultDict = pf.lossfunction(proteinID, predictions, Y_test)

# Calcuations done, record time, then calculate duration
endTime=time.clock()
duration=endTime-startTime

