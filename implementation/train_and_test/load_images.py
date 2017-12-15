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
from sklearn import model_selection
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
import math
v_1 = True
from ast import literal_eval




def store_entire_equation(path, folders, start, end, name):
    hw_objects = read_equations(folders, start, end)
    storage_directory = path + "{}_equations/".format(name)
    print storage_directory

    if not os.path.exists(storage_directory):
        os.makedirs(storage_directory)
    for index, hw in enumerate(hw_objects):
        print "Done with #{}".format(index)
        hw.get_training_example_v3(storage_directory)


from operator import itemgetter

import util
def add_to_test_v4():
    return random.random() < 0.10


def store_images(path, folders, start, end, name):
    hw_objects = read_equations(folders, start, end)
    storage_directory = path + name
    print storage_directory
    if not os.path.exists(storage_directory):
        os.makedirs(storage_directory)

    for index, hw in enumerate(hw_objects):
        hw.get_training_example_v2(storage_directory)
        print "Done with #{}".format(index)

def store_info(path, folders, start, end, name):
    # Idea: store sparse dictionary
    hw_objects = read_equations(folders, start, end)
    storage_directory = path + name
    if not os.path.exists(storage_directory):
        os.makedirs(storage_directory)

    for index, hw in enumerate(hw_objects):
        print "Done with #{}".format(index)
        x,y = hw.get_training_example_without_hog()
        rep_normal = {}
        for np_array, symbol in zip(x, y):
            # Convert sparse np array to dictionary

            rows, cols = np_array.shape
            rep_normal['num_rows'] = rows
            rep_normal['num_cols'] = cols
            np_array_map = {}
            for row in range(rows):
                for col in range(cols):
                    if np_array[row][col] != 0:
                        np_array_map[str((row, col))] = np_array[row][col]

            if 'symbol' not in rep_normal:
                rep_normal['symbol'] = [[symbol, np_array_map]]
            else:
                rep_normal['symbol'].append([symbol, np_array_map])


        filename_normal = storage_directory + hw.filename
        with open(filename_normal, "w") as fp:
            cPickle.dump(rep_normal, fp)



results = []
import matplotlib.pyplot as plt

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

        plt.imshow(np_array,cmap='gray')
        #plt.savefig("results/{}".format(symbol[0]))
        TRAINING_X.append(list(itertools.chain.from_iterable(np_array)))

   #print "Done with read"
    results.append((TRAINING_X, TRAINING_Y))

from PIL import Image
from numpy import array
from scipy.ndimage.filters import gaussian_filter
def load_info_v2(short_filename, equation_file, downsize=False):
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


        if downsize:
            # plt.imshow(np_array, cmap='gray')
            # plt.savefig("test_{}".format(symbol[0]))
            image = Image.fromarray(np_array)
            image = image.resize((32,32),Image.ANTIALIAS)
            # image.show()
            np_array = array(image)


        dir = '/Users/norahborus/Documents/DATA/training_data/CROHME_image_attempt/'
        filename = dir + "_" + short_filename + symbol[0]+ '.png'
        im = Image.fromarray(np_array * 255.0)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(filename, "PNG")
       # np_array = gaussian_filter(np_array, sigma=7)
        #img.resize((100,100),Image.ANTIALIAS)
       # img.show()
        #img.save("test.jpg")

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
    return random.random() < 0.01

def partial_fit(TRAINING_X, TRAINING_Y, mean, std):
    print "********partial fit*********"
    #alpha = 1.0 / (len(TRAINING_Y)*50)
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

   # tuning_parameters = {'alpha': [10**a for a in range(-6,-2)]}
   # clf = model_selection.GridSearchCV(linear_model.SGDClassifier(loss='hinge', penalty='elasticnet', l1_ratio=0.15, n_iter=5, shuffle=True,verbose=False,
   #                                                           n_jobs=10, average=False,class_weight='balanced'),
   #                                                           tuning_parameters,cv=10, scoring='f1_macro')
    for n in range(1):
        print "iteration #{}".format(n+1)
        random.shuffle(shuffled_range)
        for i in shuffled_range:
            batch_x, batch_y = [TRAINING_X[i]], [TRAINING_Y[i]]
            training_data = np.fromiter(chain.from_iterable([(val.toarray()).flatten() for j, val in enumerate(batch_x)]), np.float).reshape(len(batch_x), 40000)
            #Normalizing
          #  training_data -= mean
          #  training_data /= std
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

def load_info_v3(equation_file):
   # print "***********************************************"
    TRAINING_X = []
    TRAINING_Y = []

    with open(equation_file, 'rb') as fp:
        rep = cPickle.load(fp)

    symbols = rep['symbol']
    for symbol in symbols:
        TRAINING_Y.append(symbol[0])
        np_array_map = symbol[1]
        rows,cols = rep['num_rows'], rep['num_cols']
        np_array = np.zeros((rows,cols))
        for row in xrange(rows):
            for col in xrange(cols):
                if str((row,col)) in np_array_map:
                    np_array[row][col] = np_array_map[str((row,col))]

        image = Image.fromarray(np_array*255)
        image = image.resize((32,32),Image.ANTIALIAS)
        image_2 = image.convert('RGB')
        image_2.save('results/'+symbol[0], 'png')
        break
        TRAINING_X.append(image)

    return TRAINING_X, TRAINING_Y



import util
import collections



from data_augmentation import combined_with_augmentation_limit_500


def read_image_files_v4(folders):
    path = folders[0]
    TRAINING_X = []
    TRAINING_Y = []
    TEST_X = []
    TEST_Y = []
    y_map = {}
    counter = 0
    train_filenames = np.array(os.listdir(path + folders[1]))
    np.random.shuffle(train_filenames)

    train_y_freq = defaultdict(int)
    test_y_freq = defaultdict(int)
    for index, filename in enumerate(train_filenames):
        x = path + folders[1] + filename
        if 'kml' not in filename:
            continue

        if 'png' not in filename:
            y = filename[filename.index('kml')+3:]
            if  y == 'forward_slash':
                y = '/'
        else:
            y = filename[filename.index('kml')+3:-4]
            if  y == 'forward_slash':
                y = '/'

        if y not in y_map:
            y_map[y] = counter
            counter += 1

        if add_to_test_v4():
            test_y_freq[y] += 1
            TEST_X.append(x)
            TEST_Y.append(y)
        else:
            train_y_freq[y] += 1
            TRAINING_X.append(x)
            TRAINING_Y.append(y)


    test_y_freq = collections.OrderedDict(sorted(test_y_freq.items()))
    print "TEST Y FREQ: ", test_y_freq
    print len(TRAINING_Y), len(TEST_Y)
    return TRAINING_X, TRAINING_Y, TEST_X, TEST_Y, y_map

def read_image_files_v2(folders, segment = False):
    path = folders[0]
    TRAINING_X = []
    TRAINING_Y = []
    TEST_X = []
    TEST_Y = []
    y_map = {}
    counter = 0
    train_filenames = np.array(os.listdir(path + folders[1]))
    test_filenames = np.array(os.listdir(path + folders[2]))
    test_y_freq = defaultdict(int)
    train_y_freq = defaultdict(int)
   # np.random.shuffle(train_filenames)
   # np.random.shuffle(test_filenames)
    print util.commonly_missegmented_symbols
    for filename in test_filenames[:400]:
        x = path + folders[2] +filename
        if 'kml' not in filename:
            continue


        if 'png' not in filename:
            y = filename[filename.index('kml')+3:]
            if  y == 'forward_slash':
                y = '/'
        else:
            y = filename[filename.index('kml')+3:filename.index('png')-1]
            if  y == 'forward_slash':
                y = '/'
           # print "Filename: {}, symbol: {}".format(x, y)



        if segment:
            print y
            if y in util.commonly_missegmented_symbols:
                if y not in y_map:
                    y_map[y] = counter
                    counter += 1
                TEST_X.append(x)
                TEST_Y.append(y)

        else:
            test_y_freq[y] += 1
            if y not in y_map:
                y_map[y] = counter
                counter += 1

            TEST_X.append(x)

            TEST_Y.append(y)

    print "Done with test files"

    train_y = set(TRAINING_Y)
    test_y = set(TEST_Y)

    diff = train_y - test_y
    for index, filename in enumerate(train_filenames[:4000]):
      # print "Train file: {}".format(index)
        x = path + folders[1] +filename
        if 'kml' not in filename:
            continue

        if 'png' not in filename:
            y = filename[filename.index('kml')+3:]
            if  y == 'forward_slash':
                y = '/'
        else:
            y = filename[filename.index('kml')+3:-4]
            if  y == 'forward_slash':
                y = '/'
           # print "Filename: {}, symbol: {}".format(x, y)
        if segment:
           # print y
            if y in util.commonly_missegmented_symbols:
                if y not in y_map:
                    y_map[y] = counter
                    counter += 1
                TRAINING_X.append(x)
                TRAINING_Y.append(y)

        else:
            train_y_freq[y] += 1
            if y not in y_map:
                y_map[y] = counter
                counter += 1

            TRAINING_X.append(x)
            TRAINING_Y.append(y)

    train_y_freq = collections.OrderedDict(sorted(train_y_freq.items()))
    test_y_freq = collections.OrderedDict(sorted(test_y_freq.items()))
    print "Difference: {}".format(diff)
    print "Frequency: TRAIN {}".format(train_y_freq)
    print "Frequency: TEST {}".format(test_y_freq)

    print len(TRAINING_Y), len(TEST_Y)
    return TRAINING_X, TRAINING_Y, TEST_X, TEST_Y, y_map

from PIL import Image
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
from PIL import Image
from skimage import data, color, exposure
from data_augmentation import combined_with_augmentation_limit_500


def read_images_flattened_v2(folders, segment = False, using_hog_features = False):
    path = folders[0]
    TRAINING_X = []
    TRAINING_Y = []
    TEST_X = []
    TEST_Y = []
    y_map = {}
    counter = 0
    train_filenames = np.array(os.listdir(path + folders[1]))
    test_filenames = np.array(os.listdir(path + folders[2]))

    train_y_freq = defaultdict(int)
    test_y_freq = defaultdict(int)

  #  np.random.shuffle(train_filenames)
  #  np.random.shuffle(test_filenames)

    print util.commonly_missegmented_symbols

    for index, filename in enumerate(train_filenames[:5000]):
        print "Train file: {}".format(index)
        if 'kml' not in filename:
          continue
        x = path + folders[1] +filename
        if using_hog_features:
            image = color.rgb2gray(np.asarray((Image.open(x)).convert('RGB')))
            fd, hog_image = hog(image, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(1, 1), visualise=True)

            # Rescale histogram for better display
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
            array = (np.asarray(hog_image_rescaled)).flatten()
        else:
            image = Image.open(x).convert('L')
            array = (np.asarray(image)).flatten()

        if 'png' not in filename:
            y = filename[filename.index('kml')+3:]
            if  y == 'forward_slash':
                y = '/'
        else:
            y = filename[filename.index('kml')+3:-4]
            if  y == 'forward_slash':
                y = '/'
        if y not in y_map:
            y_map[y] = counter
            counter += 1

        if y in util.commonly_missegmented_symbols:
            if add_to_test_v4():
                test_y_freq[y] += 1
                TEST_X.append(array)
                TEST_Y.append(y)
            else:
                train_y_freq[y] += 1
                TRAINING_X.append(array)
                TRAINING_Y.append(y)
        else:
            TRAINING_X.append(array)
            TRAINING_Y.append(y)


    print len(TRAINING_Y), len(TEST_Y)
    return TRAINING_X, TRAINING_Y, TEST_X, TEST_Y, y_map



def read_images_flattened(folders, segment = False, using_hog_features = False, train_bound=None, test_bound=None):
    path = folders[0]
    TRAINING_X = []
    TRAINING_Y = []
    TEST_X = []
    TEST_Y = []
    y_map = {}
    counter = 0
    if folders[1] == folders[2]:
        print "Reduced size"
        all_files = np.array(os.listdir(path + folders[1]))
        np.random.shuffle(all_files)
        all_files = all_files[:1000]
        m, = all_files.shape
        print "M: ", m
        test_filenames = all_files[:m / 10]
        train_filenames = all_files[m / 10:]
        print len(test_filenames), len(train_filenames)

    else:
        train_filenames = np.array(os.listdir(path + folders[1]))
        test_filenames = np.array(os.listdir(path + folders[2]))

        np.random.shuffle(train_filenames)
        np.random.shuffle(test_filenames)

    if train_bound != None:
        train_filenames = train_filenames[:train_bound]

    if test_bound != None:
        test_filenames = test_filenames[:test_bound]

    for index, filename in enumerate(test_filenames):
        print "*********************************TEST FILE #{}".format(index)
        x = path + folders[2] +filename
        if 'kml' not in filename:
            continue

        if using_hog_features:
            image = color.rgb2gray(np.asarray((Image.open(x)).convert('RGB')))
            fd, hog_image = hog(image, orientations=8, pixels_per_cell=(32, 32),cells_per_block=(1, 1), visualise=True)
            # Rescale histogram for better display
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
            array = (np.asarray(hog_image_rescaled)).flatten()
        else:
            image = Image.open(x).convert('L')
            array = np.asarray(image)
            array = array / 255.0
           # print "SHAPE: {}".format(array.shape)
            array = array.flatten()


        if 'png' not in filename:
            y = filename[filename.index('kml')+3:]
            if  y == 'forward_slash':
                y = '/'
        else:
            y = filename[filename.index('kml')+3:-4]
            if  y == 'forward_slash':
                y = '/'


        if y not in y_map:
            y_map[y] = counter
            counter += 1


        TEST_X.append(array)
        TEST_Y.append(y)

    print "Done with test files"

    for index, filename in enumerate(train_filenames):
        print "*********************************TRAIN FILE #{}".format(index)
        if 'kml' not in filename:
          continue
        x = path + folders[1] + filename
        if using_hog_features:
            print "using HOG"
            image = color.rgb2gray(np.asarray((Image.open(x)).convert('RGB')))
            fd, hog_image = hog(image, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(1, 1), visualise=True)

            # Rescale histogram for better display
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
            array = (np.asarray(hog_image_rescaled)).flatten()
        else:
            print "HERE HERE HERE"
            image = Image.open(x).convert('L')

            array = np.asarray(image)
            array = array / 255.0
          #  print "SHAPE: {}".format(array.shape)
            array = array.flatten()

        if 'png' not in filename:
            y = filename[filename.index('kml')+3:]
            if  y == 'forward_slash':
                y = '/'
        else:
            y = filename[filename.index('kml')+3:-4]
            if  y == 'forward_slash':
                y = '/'
           # print "Filename: {}, symbol: {}".format(x, y)

        if y not in y_map:
            y_map[y] = counter
            counter += 1

        TRAINING_X.append(array)
        TRAINING_Y.append(y)


    print len(TRAINING_Y), len(TEST_Y)
    print "Shape: ", len(TEST_X[0])
    return TRAINING_X, TRAINING_Y, TEST_X, TEST_Y, y_map



def read_image_files(folders, start, end):
    TRAINING_X = []
    TRAINING_Y = []
    TEST_X = []
    TEST_Y = []
    y_map = {}
    counter = 0
    filenames = os.listdir('results_'+str(start)+str(end)+'/')
    for filename in filenames:
        x = 'results_'+str(start)+str(end)+'/'+filename
        if 'kml' not in filename:
            continue

        if 'png' not in filename:
            y = filename[filename.index('kml')+3:]
            if  y == 'forward_slash':
                y = '/'
        else:
            y = filename[filename.index('kml')+3:-4]
            if  y == 'forward_slash':
                y = '/'
           # print "Filename: {}, symbol: {}".format(x, y)
        if y not in y_map:
            y_map[y] = counter
            counter+=1

        if add_to_test_set():
            TEST_X.append(x)
            TEST_Y.append(y)

        TRAINING_X.append(x)
        TRAINING_Y.append(y)

    print len(TRAINING_Y), len(TEST_X)
    return TRAINING_X, TRAINING_Y, TEST_X, TEST_Y, y_map


def read_images_direct(folders, start, end):
    TRAINING_X = []
    TRAINING_Y = []
    TEST_X = []
    TEST_Y = []
    y_map = {}
    counter = 0
    hw_objects = read_equations(folders, start, end)
    for hw in hw_objects:
        x, y = hw.get_training_example_v2()
        for x,y in zip(x,y):
            print "Filename: {}, symbol: {}".format(x, y)
            if y not in y_map:
                y_map[y] = counter
                counter+=1

            if add_to_test_set():
                TEST_X.append(x)
                TEST_Y.append(y)
            else:
                TRAINING_X.append(x)
                TRAINING_Y.append(y)

    print len(TRAINING_Y), len(TEST_X)
    return TRAINING_X, TRAINING_Y, TEST_X, TEST_Y, y_map

def read_tensor_convert(path):
    global v_1
    print path
    filenames = os.listdir(path)
    #  print filenames
    filenames = [path + file for file in filenames[:100]]
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
            test_data.append(load_info_v3(file))
        elif index % 10 == 0:
            bucket_0.append(load_info_v3(file))
        elif index % 10 == 1:
            bucket_1.append(load_info_v3(file))
        elif index % 10 == 2:
            bucket_2.append(load_info_v3(file))
        elif index % 10 == 3:
            bucket_3.append(load_info_v3(file))
        elif index % 10 == 4:
            bucket_4.append(load_info_v3(file))
        elif index % 10 == 5:
            bucket_5.append(load_info_v3(file))
        elif index % 10 == 6:
            bucket_6.append(load_info_v3(file))
        elif index % 10 == 7:
            bucket_7.append(load_info_v3(file))
        elif index % 10 == 8:
            bucket_8.append(load_info_v3(file))
        elif index % 10 == 9:
            bucket_9.append(load_info_v3(file))

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

    TRAINING_X = []
    TRAINING_Y = []
    TEST_X = []
    TEST_Y = []

    y_map = {}
    counter = 0
    for x,y in data:
        for val_x,val_y in zip(x,y):
            if val_y in y_map:
                continue
            y_map[val_y] = counter
            counter+=1
        print x
        TRAINING_X += x
        TRAINING_Y += y

    for x,y in test_data:
        for val_x,val_y in zip(x,y):
            if val_y in y_map:
                continue
            y_map[val_y] = counter
            counter+=1
        TEST_X += x
        TEST_Y += y


    with open("y_map.txt", "w") as fp:
        cPickle.dump(y_map, fp)

    return TRAINING_X, TRAINING_Y, TEST_X, TEST_Y, y_map


def read_files_svm(path, dataset_size=None, is_test=False, is_train=False):
    filenames = os.listdir(path)
    filenames = [file for file in filenames[:20]]
    start = datetime.datetime.now()
    X = []
    Y = []
    gc.disable()
    if dataset_size != None:
        filenames = filenames[:dataset_size]

    print "TOTAL # FILES FOR PATH", path,  len(filenames)
    for index, file in enumerate(filenames):
        if 'inkml' not in file:
            continue
        print "Done --> {}".format(index)
        x, y = load_info_v2(file, path + file)
        X += x
        Y += y
        if is_train and len(Y) >= 6000:
            break
        if is_test and len(Y) >= 600:
            break


    gc.enable()

    if is_test:
        Y = Y[:600]
        X = X[:600]
    else:
        Y = Y[:6000]
        X = X[:6000]

    print len(Y)
   # X = [(val.toarray()).flatten() for i, val in enumerate(X)]
    end = datetime.datetime.now()
    time_taken = (end - start).total_seconds()
    print "TOTAL TIME TAKEN: {}".format(time_taken)
    return X, Y

def read_files_sequence(path, folders):
    global v_1
    train_dir = path + folders[1]
    test_dir = path + folders[2]

    print "Directories: ", train_dir, test_dir
    TRAINING_X, TRAINING_Y = read_files_svm(train_dir, is_train=True)
    TEST_X, TEST_Y = read_files_svm(test_dir, is_test=True)

    if v_1:
        TRAINING_X = [(val.toarray()).flatten() for i, val in enumerate(TRAINING_X)]
        TEST_X = [(val.toarray()).flatten() for i, val in enumerate(TEST_X)]
        '''
        sparse_test_data = csr_matrix((np.fromiter(
            chain.from_iterable([(val.toarray()).flatten() for i, val in enumerate(TEST_X)]), np.float)).reshape(
            len(TEST_Y), 1024))

        sparse_training_data = csr_matrix((np.fromiter(
            chain.from_iterable([(val.toarray()).flatten() for i, val in enumerate(TRAINING_X)]), np.float)).reshape(
            len(TRAINING_Y), 1024))
        '''
        print len(TRAINING_X[0])
        return TRAINING_X, TRAINING_Y, TEST_X, TEST_Y

    else:
        mean = 0
        std = 0
        clf, y_map = partial_fit(TRAINING_X, TRAINING_Y,mean, std)
        "Done with partial fit"
        return clf, y_map, TEST_X, TEST_Y


   # store = first_path + '12_dump.txt'
   # with open(store, 'w') as fp:
   #     cPickle.dump(data, fp)

def read_files_sequence_v2(path, folders):
    global v_1
    dir = path + folders[1]
    X, Y = read_files_svm(dir)

    X = [(val.toarray()).flatten() for i, val in enumerate(X)]
    '''
    sparse_test_data = csr_matrix((np.fromiter(
        chain.from_iterable([(val.toarray()).flatten() for i, val in enumerate(TEST_X)]), np.float)).reshape(
        len(TEST_Y), 1024))

    sparse_training_data = csr_matrix((np.fromiter(
        chain.from_iterable([(val.toarray()).flatten() for i, val in enumerate(TRAINING_X)]), np.float)).reshape(
        len(TRAINING_Y), 1024))
    '''
    print len(X[0])
    data = np.array(X)
    m, n = data.shape
    labels = (np.array(Y)).reshape((m, 1))
    p = np.random.permutation(m)
    data = data[p, :]
    labels =labels[p, :]
    TEST_X = data[0:m / 10, :]
    TEST_Y= labels[0:m / 10, :]
    TRAINING_X = data[m / 10:, :]
    TRAINING_Y = labels[m / 10:, :]

    TRAINING_Y = TRAINING_Y.reshape((TRAINING_Y.shape[0],))
    TEST_Y = TEST_Y.reshape((TEST_Y.shape[0],))
    return TRAINING_X, TRAINING_Y, TEST_X, TEST_Y

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
    poster_session = True
    if poster_session:
        '''
        path = '/Users/norahborus/Documents/DATA/training_data/POSTER_SESSION/'
        training_folders = ['/Users/norahborus/Documents/DATA/training_data/POSTER_SESSION/', "23/"]
       # training_folders = ['/Users/norahborus/Documents/DATA/training_data/POSTER_SESSION/', "12_without_hog/"]
        TRAINING_X, TRAINING_Y, TEST_X, TEST_Y = read_files_sequence_v2(path, training_folders)
        '''
        path = '/Users/norahborus/Documents/DATA/training_data/'
        training_folders = ['/Users/norahborus/Documents/DATA/training_data/', "CROHME_half_size/"]
        # training_folders = ['/Users/norahborus/Documents/DATA/training_data/POSTER_SESSION/', "12_without_hog/"]
        TRAINING_X, TRAINING_Y, TEST_X, TEST_Y = read_files_sequence_v2(path, training_folders)
        print "Number of train examples: ", len(TRAINING_Y), "Number of test examples: ", len(TEST_Y)
        y_map = {}
        counter = 0
        NEW_TRAINING_Y = []
        weights = []
        freq = defaultdict(int)
        # print TRAINING_Y
        for y in TRAINING_Y:
          #  print y, type(y)
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

        for y in TEST_Y:
            if y in y_map:
                continue
            y_map[y] = counter
            counter += 1

        #  print len(X), len(Y)
        # for example, y in zip(X, Y):
        #     print len(example), y
        print "About to start fitting"
        clf = svm.LinearSVC(multi_class='ovr', C=50.0)
        clf.fit(TRAINING_X, TRAINING_Y, weights)
        print "***AFTER*****"
        # hps, _, _ = optunity.maximize(svm_auc, num_evals=200, logC=[-5, 2], logGamma=[-5, 1])
        # clf = svm.SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma']).fit(X, Y)

        with open("SVMs/svm_CROHME", "w") as fp:
            cPickle.dump(clf, fp)

        correct = 0
        error = 0
        for i in range(len(TEST_X)):
            dec = clf.decision_function([TEST_X[i]])
            max_index = np.argmax(dec[0])
         #   print "Len dec: {}, Dec: {}".format(len(dec[0]), dec[0])
           # print "Max: {}".format(max(dec[0]))
          #  print "Max index: {}".format(max_index)
            for symbol, index in y_map.iteritems():
                if index == max_index:
                  #  print "Matching symbol: {}, Truth: {}".format(symbol, TEST_Y[i])
                    if symbol != TEST_Y[i]:
                        error += 1
                    else:
                        correct += 1

        print "TEST Accuracy: {}".format(1.0 * correct / len(TEST_Y))
        print "TEST Error: {}".format(1.0 * error / len(TEST_Y))

    if not poster_session and v_1:

        path = '/Users/norahborus/Documents/DATA/COMBINED/'
        training_folders = ['/Users/norahborus/Documents/DATA/COMBINED/', 'TRAIN/', 'TEST/']
        path = '/Users/norahborus/Documents/DATA/training_data/'
        training_folders = ['/Users/norahborus/Documents/DATA/training_data/', "12/", "23/"]
        training_folders = ['/Users/norahborus/Documents/DATA/training_data/', "CROHME_training_2011/",
                            "TrainINKML_2013/", "trainData_2012_part1/", "trainData_2012_part2/"]
        training_hw_folders = ['/Users/norahborus/Documents/DATA/COMBINED/', 'TEST_preprocessed/',
                               'TRAIN_preprocessed/']

        TRAINING_X, TRAINING_Y, TEST_X, TEST_Y = read_files_sequence(path, training_folders)

        trainData = np.array(TRAINING_X)
        m, n = trainData.shape
        trainLabels = (np.array(TRAINING_Y)).reshape((m,1))

        print "Shape:", trainData.shape
        print "Shape: ", trainLabels.shape
        p = np.random.permutation(m)
        trainData = trainData[p, :]
        trainLabels = trainLabels[p, :]
        devData = trainData[0:m / 10, :]
        devLabels = trainLabels[0:m / 10, :]
        trainData = trainData[m / 10:, :]
        trainLabels = trainLabels[m / 10:, :]

        print trainLabels.shape, devLabels.shape
        dev_shape = devLabels.shape
        train_shape = trainLabels.shape

        trainLabels = trainLabels.reshape((train_shape[0],))
        devLabels = devLabels.reshape((dev_shape[0],))


        TRAINING_X = trainData
        TRAINING_Y = trainLabels

        DEV_X = devData
        DEV_Y = devLabels

        print "Shape: ", TRAINING_Y.shape, TRAINING_X.shape,  DEV_Y.shape, DEV_X.shape
        print "Ordering TRAINING_Y:"

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

        TRAINING_Y = NEW_TRAINING_Y

        for y in TEST_Y:
            if y in y_map:
                continue
            y_map[y] = counter
            counter += 1



        #  print len(X), len(Y)
        # for example, y in zip(X, Y):
        #     print len(example), y
        print "About to start fitting"
        print len(TRAINING_Y), len(DEV_Y), len(TEST_Y)

        '''
        clf = svm.SVC(decision_function_shape='ovo', gamma=0.01, C=100.0)
        clf.fit(TRAINING_X, TRAINING_Y, weights)
        clf.decision_function_shape = "ovr"
        '''

        clf = svm.LinearSVC(multi_class='ovr', C=50.0)
        clf.fit(TRAINING_X, TRAINING_Y, weights)
        print "***AFTER*****"
        # hps, _, _ = optunity.maximize(svm_auc, num_evals=200, logC=[-5, 2], logGamma=[-5, 1])
        # clf = svm.SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma']).fit(X, Y)

        error = 0

        correct = 0
        for i in range(len(DEV_X)):
            dec = clf.decision_function([DEV_X[i]])
            max_index = np.argmax(dec[0])
          #  print "Len dec: {}, Dec: {}".format(len(dec[0]), dec[0])
           # print "Max: {}".format(max(dec[0]))
          #  print "Max index: {}".format(max_index)
            for symbol, index in y_map.iteritems():
                if index == max_index:
                  #  print "Matching symbol: {}, Truth: {}".format(symbol, TRAINING_Y[i])
                    if symbol != DEV_Y[i]:
                        error += 1
                    else:
                        correct += 1

        print "DEV Accuracy: {}".format(1.0 * correct / len(DEV_Y))
        print "DEV Error: {}".format(1.0 * error / len(DEV_Y))


        correct = 0
        error = 0
        for i in range(len(TEST_X)):
            dec = clf.decision_function([TEST_X[i]])
            max_index = np.argmax(dec[0])
         #   print "Len dec: {}, Dec: {}".format(len(dec[0]), dec[0])
           # print "Max: {}".format(max(dec[0]))
          #  print "Max index: {}".format(max_index)
            for symbol, index in y_map.iteritems():
                if index == max_index:
                  #  print "Matching symbol: {}, Truth: {}".format(symbol, TEST_Y[i])
                    if symbol != TEST_Y[i]:
                        error += 1
                    else:
                        correct += 1

        print "TEST Accuracy: {}".format(1.0 * correct / len(TEST_Y))
        print "TEST Error: {}".format(1.0 * error / len(TEST_Y))

    elif not poster_session:
        clf, y_map, TEST_X, TEST_Y = read_files_sequence(path, path + "12_modified/")
        with open("y_map.txt","w") as fp:
            cPickle.dump(y_map, fp)

        error = 0
        for i in range(len(TEST_X)):
            entry = ((TEST_X[i].toarray()).flatten()).reshape(1,40000)
            dec = clf.decision_function(entry)
            print "Max: {}".format(max(dec[0]))
            max_index = np.argmax(dec[0])

            for symbol, index in y_map.iteritems():
                if index == max_index:
                    print "Matching symbol: {}, Truth: {}".format(symbol, TEST_Y[i])
                    if symbol != TEST_Y[i]:
                        error += 1

        print "TEST Error: {}".format(1.0 * error / len(TEST_Y))

def store():
    path = '/Users/norahborus/Documents/DATA/training_data/'
    training_folders = ['/Users/norahborus/Documents/DATA/training_data/', "CROHME_training_2011/",
                        "TrainINKML_2013/", "trainData_2012_part1/", "trainData_2012_part2/"]
    store_info(path, training_folders, 1,2, "12_without_hog/")



if __name__ == '__main__':
    random.seed(10)
    #path = '/Users/norahborus/Documents/DATA/training_data/'
   # training_folders = ['/Users/norahborus/Documents/DATA/training_data/', "CROHME_training_2011/", "TrainINKML_2013/", "trainData_2012_part1/", "trainData_2012_part2/"]
   # read_images_direct(training_folders, 1,2)
  #  store()
    main()

