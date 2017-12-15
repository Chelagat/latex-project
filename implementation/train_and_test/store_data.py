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
import os


def store_equations_with_half_image_size(path, folders, start, end, name):
    # Idea: store sparse dictionary
    hw_objects = read_equations(folders, start, end)
    storage_directory = path + name
    if not os.path.exists(storage_directory):
        os.makedirs(storage_directory)

    for index, hw in enumerate(hw_objects):
        print "Done with #{}".format(index)
        x,y = hw.get_training_example_half_size()
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

def store_images_half_size(path, folders, start, end, name):
    hw_objects = read_equations(folders, start, end)
    storage_directory = path + name
    print storage_directory
    if not os.path.exists(storage_directory):
        os.makedirs(storage_directory)

    for index, hw in enumerate(hw_objects):
        hw.get_training_example_half_size_v2(storage_directory)
        print "Done with #{}".format(index)


def store_images(path, folders, start, end, name):
    hw_objects = read_equations(folders, start, end)
    storage_directory = path + name
    print storage_directory
    if not os.path.exists(storage_directory):
        os.makedirs(storage_directory)

    for index, hw in enumerate(hw_objects):
        hw.get_training_example_v2(index, storage_directory)
        print "Done with #{}".format(index)

def store_images_quarter_size(path, folders, start, end, name):
    hw_objects = read_equations(folders, start, end)
    storage_directory = path + name
    print storage_directory
    if not os.path.exists(storage_directory):
        os.makedirs(storage_directory)

    for index, hw in enumerate(hw_objects):
        hw.get_training_example_v2(index, storage_directory)
        print "Done with #{}".format(index)


if __name__ == '__main__':
   # path = '/Users/norahborus/Documents/DATA/training_data/'
    training_folders = ['/Users/norahborus/Documents/DATA/training_data/original/', "CROHME_training_2011/",
                        "TrainINKML_2013/", "trainData_2012_part1/", "trainData_2012_part2/"]
   # store_equations_with_half_image_size(path, training_folders, 2,3, "TrainINKML_half_size/")
  #  store_images_half_size(path, training_folders, 2,3, 'TrainINKML_images_half_size/')
    path = '/Users/norahborus/Documents/DATA/training_data/200x200/'
    store_images_quarter_size(path, training_folders, 1,2, 'CROHME_Characters/')