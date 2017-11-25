import thread
import time
import pickle
import os
import json
from ast import literal_eval
import numpy as np
import itertools
from inkml import svm_train
from inkml import svm_linear_train
from inkml import sparse_svm_linear_train
import threading
from multiprocessing import Pool
import datetime
import cPickle
from inkml import read_equations
import gc
import marshal
from scipy.sparse import csr_matrix
import scipy.sparse

from itertools import chain
from sklearn import svm
from sklearn import linear_model
import json
import signal
import sys
import logging
import matplotlib
from sys import argv
matplotlib.use("Agg")
import itertools
#from store_numpy_array import load_info
from collections import defaultdict
import datetime
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)
from scipy.sparse import csr_matrix

import random
from xml.dom.minidom import parseString

# hwrt modules
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import handwritten_data

v_1 = False

def store_info(path, folders, start, end):
    '''
    Update: stored 100 equations so far in crohme, start running from file 101
    '''
    # Idea: store sparse dictionary
    hw_objects = read_equations(folders, start, end)
    storage_directory = path + str(start) + str(end) + '/'

    if not os.path.exists(storage_directory):
        os.makedirs(storage_directory)

    for index, hw in enumerate(hw_objects):
        print "Done with #{}".format(index)
        x, y = hw.get_training_example()
        rep = {}

        for np_array, symbol in zip(x, y):
            # Convert sparse np array to dictionary
            #  print symbol
            rows, cols = np_array.shape
            rep['num_rows'] = rows
            rep['num_cols'] = cols
            np_array_map = {}
            for row in range(rows):
                for col in range(cols):
                    if np_array[row][col] != 0:
                        np_array_map[str((row, col))] = np_array[row][col]

            if 'symbol' not in rep:
                rep['symbol'] = [[symbol, np_array_map]]
            else:
                rep['symbol'].append([symbol, np_array_map])

        filename = storage_directory + hw.filename

        with open(filename, "w") as fp:
            cPickle.dump(rep, fp)

    '''
    storage_directory_svm = storage_directory + "svm/"
    if not os.path.exists(storage_directory_svm):
        os.makedirs(storage_directory_svm)

    filename = storage_directory_svm + "svm.pickle"
    with open(filename, "w") as fp:
        pickle.dump(svm, fp)

    '''



results = []
def load_info(equation_file):
    global results
   # print "***********************************************"
    TRAINING_X = []
    TRAINING_Y = []
    with open(equation_file, 'rb') as fp:
        rep = cPickle.load(fp)

    symbols = rep['symbol']
    for symbol in symbols:
      #  print symbol[0]
        TRAINING_Y.append(symbol[0])
        np_array_map = symbol[1]
        rows,cols = rep['num_rows'], rep['num_cols']
        np_array = np.zeros((rows,cols))
        for row in xrange(rows):
            for col in xrange(cols):
                if str((row,col)) in np_array_map:
                    np_array[row][col] = np_array_map[str((row,col))]

        #plt.imshow(np_array,cmap='gray')
        #plt.savefig("results/{}".format(symbol[0]))
        TRAINING_X.append(list(itertools.chain.from_iterable(np_array)))

   #print "Done with read"
    results.append((TRAINING_X, TRAINING_Y))

def load_info_v2(equation_file):
   # print "***********************************************"
    TRAINING_X = []
    TRAINING_Y = []


    with open(equation_file, 'rb') as fp:
        rep = cPickle.load(fp)

    symbols = rep['symbol']
    for symbol in symbols:
      #  print symbol[0]
        TRAINING_Y.append(symbol[0])
        np_array_map = symbol[1]
        rows,cols = rep['num_rows'], rep['num_cols']
        np_array = np.zeros((rows,cols))
        for row in xrange(rows):
            for col in xrange(cols):
                if str((row,col)) in np_array_map:
                    np_array[row][col] = np_array_map[str((row,col))]

        sparse_matrix = csr_matrix(np_array)
       # print sparse_matrix.shape
        #plt.imshow(np_array,cmap='gray')
        #plt.savefig("results/{}".format(symbol[0]))
        TRAINING_X.append(sparse_matrix)

    #print "Done with read"
    return TRAINING_X, TRAINING_Y

def read_files_thread(path):
    global results
    print path
    filenames = os.listdir(path)
    print filenames
    filenames = [path+file for file in filenames]
    threads = []
    start = datetime.datetime.now()
    for file in filenames:
        if 'Store' in file:
            continue
        t = threading.Thread(target=load_info, args=(file,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    end = datetime.datetime.now()
    time_taken = (end-start).total_seconds()
    print "TOTAL TIME TAKEN: {}".format(time_taken)
    print len(results)

def add_to_test_set():
    return random.random() < 0.1

def partial_fit(TRAINING_X, TRAINING_Y):
    print "********partial fit*********"
    clf = linear_model.SGDClassifier()
    total = len(TRAINING_Y)
    step_size = total / 10
    y_map = {}
    counter = 0
    freq = defaultdict(int)
    NEW_TRAINING_Y = []
    weights = []

    for y in TRAINING_Y:
        freq[y] += 1
        if y in y_map:
            NEW_TRAINING_Y.append(y_map[y])
            continue
        NEW_TRAINING_Y.append(counter)
        y_map[y] = counter
        counter += 1

    for y in TRAINING_Y:
        weights.append(1.0 / freq[y])

    TRAINING_Y = NEW_TRAINING_Y
    shuffled_range = range(len(TRAINING_Y))
    for n in range(5):
        print "iteration #{}".format(n+1)
        random.shuffle(shuffled_range)
        for i in shuffled_range:
            batch_x, batch_y = [TRAINING_X[i]], [TRAINING_Y[i]]
            training_data = np.fromiter(chain.from_iterable([(val.toarray()).flatten() for j, val in enumerate(batch_x)]), np.float).reshape(len(batch_x), 307200)
          #  print "shape of data: {}, shape of weights: {}".format(training_data.shape,len([shuffled_weights[i]]))
            #  print "weights: {}".format(shuffled_weights[i:i+step_size])
            #  print "*************************************"
            clf.partial_fit(training_data, batch_y, classes=np.unique(TRAINING_Y),
                            sample_weight=[weights[i]])
        '''
        for i in range(0, total, step_size):
           # print "***********************************"
           # print "Value of i: {}".format(i)
            batch_x, batch_y = shuffled_x[i:i+step_size], shuffled_y[i:i+step_size]
            sparse_training_data = csr_matrix((np.fromiter(
                chain.from_iterable([(val.toarray()).flatten() for j, val in enumerate(batch_x)]), np.float)).reshape(
                len(batch_x), 307200))

          #  print "shape of data: {}, shape of weights: {}".format(sparse_training_data.shape,len(shuffled_weights[i:i+step_size]))
          #  print "weights: {}".format(shuffled_weights[i:i+step_size])
          #  print "*************************************"
            clf.partial_fit(sparse_training_data, batch_y, classes = np.unique(TRAINING_Y),sample_weight=shuffled_weights[i:i+step_size])
        '''
    return clf, y_map

def read_files_sequence(first_path, path):
    global v_1
    print path
    filenames = os.listdir(path)
    print filenames
    filenames = [path+file for file in filenames[:4000]]
    data = []
    start = datetime.datetime.now()
    bucket_0 = []
    bucket_1 = []
    bucket_2 = []
    bucket_3 = []
    bucket_4 = []
    bucket_5 = []
    bucket_6 = []
    bucket_7 = []
    bucket_8 = []
    bucket_9 = []

    test_data = []
    gc.disable()
    for index, file in enumerate(filenames):
        print "Done --> {}".format(index)
        if add_to_test_set():
            test_data.append(load_info_v2(file))
        elif index % 10 == 0:
            bucket_0.append(load_info_v2(file))

        elif index % 10 == 1:
            bucket_1.append(load_info_v2(file))
        elif index % 10 == 2:
            bucket_2.append(load_info_v2(file))
        elif index % 10 == 3:
            bucket_3.append(load_info_v2(file))
        elif index % 10 == 4:
            bucket_4.append(load_info_v2(file))
        elif index % 10 == 5:
            bucket_5.append(load_info_v2(file))
        elif index % 10 == 6:
            bucket_6.append(load_info_v2(file))
        elif index % 10 == 7:
            bucket_7.append(load_info_v2(file))
        elif index % 10 == 8:
            bucket_8.append(load_info_v2(file))
        elif index % 10 == 9:
            bucket_9.append(load_info_v2(file))

    gc.enable()
    data = []
    data += bucket_0
    data += bucket_1
    data += bucket_2
    data += bucket_3
    data += bucket_4
    data += bucket_5
    data += bucket_6
    data += bucket_7
    data += bucket_8
    data += bucket_9


   # print len(data)
    TRAINING_X = []
    TRAINING_Y = []
    for x,y in data:
        TRAINING_X += x
        TRAINING_Y += y

    TEST_X = []
    TEST_Y = []

    for x,y in test_data:
        TEST_X += x
        TEST_Y += y

    "Done sparse test data"
    if v_1:
        sparse_test_data = csr_matrix((np.fromiter(
            chain.from_iterable([(val.toarray()).flatten() for i, val in enumerate(TEST_X)]), np.float)).reshape(
            len(TEST_Y), 307200))
        print len(TRAINING_Y), len(TEST_Y)
        sparse_training_data = csr_matrix((np.fromiter(
            chain.from_iterable([(val.toarray()).flatten() for i, val in enumerate(TRAINING_X)]), np.float)).reshape(
            len(TRAINING_Y), 307200))
        print "X"
        print "Done creating test & converting to sparse"
        end = datetime.datetime.now()
        time_taken = (end - start).total_seconds()
        print "TOTAL TIME TAKEN: {}".format(time_taken)
        return sparse_training_data, TRAINING_Y, sparse_test_data, TEST_Y

    else:
        clf, y_map = partial_fit(TRAINING_X, TRAINING_Y)
        "Done with partial fit"
        return clf, y_map, TEST_X, TEST_Y


   # store = first_path + '12_dump.txt'
   # with open(store, 'w') as fp:
   #     cPickle.dump(data, fp)

def read_pickle(file):
    TRAINING_DATA = []
    with open(file, 'rb') as fp:
        TRAINING_DATA = cPickle.load(fp)

    TRAINING_X = []
    TRAINING_Y = []
    for x,y in TRAINING_DATA:
        TRAINING_X += x
        TRAINING_Y += y

    print TRAINING_X[0], TRAINING_Y[0]
    return TRAINING_X, TRAINING_Y




def read_files(path):
    print path

    filenames = os.listdir(path)
    print filenames
    filenames = [path+file for file in filenames]
    p = Pool(len(filenames))
    start = datetime.datetime.now()
    results = p.imap_unordered(load_info_v2, filenames)
    data = []
    for result in results:
        data.append(result)

    print len(data)
    end = datetime.datetime.now()
    time_taken = (end-start).total_seconds()
    print "TOTAL TIME TAKEN: {}".format(time_taken)
    store = path + '12_dump'
    with open(store, 'w') as fp:
        cPickle.dump(data, fp)

def main():
    global v_1
    path = '/Users/norahborus/Documents/DATA/training_data/'
    training_folders = ['/Users/norahborus/Documents/DATA/training_data/', "CROHME_training_2011/", "TrainINKML_2013/", "trainData_2012_part1/", "trainData_2012_part2/"]
   # store_info(path, training_folders, 2, 3)
   # print "*************SEQUENCE*****************"
    if v_1:
        sparse_training_data, TRAINING_Y, sparse_test_data, TEST_Y = read_files_sequence(path, path + "23/")
        y_map = {}
        counter = 0
        NEW_TRAINING_Y = []
        weights = []
        freq = defaultdict(int)

        print "Ordering TRAINING_Y:"
        for y in TRAINING_Y:
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

        print "About to start fitting"
        clf = svm.LinearSVC(multi_class='ovr', C=50.0)
        clf.fit(sparse_training_data, Y, weights)
        print "***AFTER*****"
        error = 0
        decisions = clf.decision_function(sparse_test_data)
        print len(decisions)
        for i, dec in enumerate(decisions):
            max_index = np.argmax(dec)
            for symbol, index in y_map.iteritems():
                if index == max_index:
                    print "Matching symbol: {}, Truth: {}".format(symbol, TEST_Y[i])
                    if symbol != TEST_Y[i]:
                        error += 1

        print "ACCURACY: SVM error: {}".format(1.0 * error / len(TEST_Y))
    else:
        clf, y_map, TEST_X, TEST_Y = read_files_sequence(path, path + "23/")
        error = 0
        for i in range(len(TEST_X)):
            entry = ((TEST_X[i].toarray()).flatten()).reshape(1,307200)
            dec = clf.decision_function(entry)
            print "Max: {}".format(max(dec[0]))
            max_index = np.argmax(dec[0])
            for symbol, index in y_map.iteritems():
                if index == max_index:
                    print "Matching symbol: {}, Truth: {}".format(symbol, TEST_Y[i])
                    if symbol != TEST_Y[i]:
                        error += 1

        print "ACCURACY: SVM error: {}".format(1.0 * error / len(TEST_Y))


if __name__ == '__main__':
    random.seed(10)
    main()
