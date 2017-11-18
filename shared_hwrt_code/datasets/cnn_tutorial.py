from inkml import read_equations
import pickle
import os
import json
from ast import literal_eval
import numpy as np
import itertools
import matplotlib.pyplot as plt
from inkml import svm_train
import random
from collections import defaultdict


from sklearn.neural_network import MLPClassifier



def cnn_train(TRAINING_X, TRAINING_Y):
    print "***********************STARTING CNN**********************************"
    TEST_X = []
    TEST_Y = []
    test_indices = random.sample(range(len(TRAINING_X)), len(TRAINING_X) / 10)
    for index in test_indices:
        TEST_X.append(TRAINING_X[index])
        TEST_Y.append(TRAINING_Y[index])

    TRAINING_X = [val for i, val in enumerate(TRAINING_X) if i not in test_indices]
    TRAINING_Y = [val for i, val in enumerate(TRAINING_Y) if i not in test_indices]

    X = TRAINING_X

    y_map = {}
    counter = 0
    NEW_TRAINING_Y = []
    weights = []
    freq = defaultdict(int)
    # print TRAINING_Y

    for y in TRAINING_Y:
        print y, type(y)
        freq[y] += 1
        if y in y_map:
            NEW_TRAINING_Y.append(y_map[y])
            continue

        NEW_TRAINING_Y.append(counter)
        y_map[y] = counter
        counter += 1

    for y in TRAINING_Y:
        weights.append(1.0 / freq[y])

    Y = NEW_TRAINING_Y

    #  print len(X), len(Y)
    # for example, y in zip(X, Y):
    #     print len(example), y


    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, learning_rate='adaptive', hidden_layer_sizes=(5, 5, 3),
                        random_state=1)
    clf.fit(X, Y)
    error = 0
    for i in range(len(TEST_X)):
        dec = clf.predict([TEST_X[i]])
        matches = []
        for val in dec:
            for symbol, index in y_map.iteritems():
                if index == val:
                    matches.append(symbol)

        print "Matching symbol(s): {}, Truth: {}".format(matches, TEST_Y[i])

  #  print "ACCURACY: SVM error: {}".format(1.0 * error / len(TEST_Y))
  #  return clf



