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
import matplotlib.pyplot as plt
from collections import Counter
#from store_numpy_array import load_info
from collections import defaultdict
import datetime
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)
from scipy.sparse import csr_matrix
import random
import os
from pprint import pprint
import json
from collections import OrderedDict
import matplotlib
import os
import shutil


def create_augment_directories():
    path = '/Users/norahborus/Documents/DATA/CLASSES/'
    symbols = ['!', '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'B', 'C', 'E',
     'F', 'G', 'H', 'I', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Y', '[', '\\Delta', '\\alpha', '\\beta', '\\cos',
     '\\div', '\\exists', '\\forall', '\\gamma', '\\geq', '\\gt', '\\in', '\\infty', '\\int', '\\lambda', '\\ldots',
     '\\leq', '\\lim', '\\log', '\\lt', '\\mu', '\\neq', '\\phi', '\\pi', '\\pm', '\\prime', '\\rightarrow', '\\sigma',
     '\\sin', '\\sqrt', '\\sum', '\\tan', '\\theta', '\\times', '\\{', '\\}', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|']

    for symbol in symbols:
        augment_dir = path + symbol + '/AUGMENTED/'
       # print augment_dir
        if not os.path.exists(augment_dir):
            print "Augment directory: ",augment_dir
            os.makedirs(augment_dir)

import yaml

def create_class_distribution():
    file = 'DATA_ANALYSIS/aggregate_class_distribution'
    with open(file, 'rb') as fp:
        freq_map = cPickle.load(fp)

    '''
    to_file = 'DATA_ANALYSIS/aggregate_class_distribution_pretty_print.yml'
    with open(to_file, 'w') as yaml_file:
     yaml.dump(freq_map, stream=yaml_file, default_flow_style=False)

    

    to_file = 'DATA_ANALYSIS/aggregate_class_distribution_pretty_print.txt'
    with open(to_file, 'w') as fp:
        for symbol in freq_map:
            fp.write("{} --> {}\n".format(symbol, freq_map[symbol]))
    '''

    print sum(freq_map.values())
def separate_images_by_class():
    '''
    keys = ['!', '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'B', 'C', 'E',
     'F', 'G', 'H', 'I', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Y', '[', '\\Delta', '\\alpha', '\\beta', '\\cos',
     '\\div', '\\exists', '\\forall', '\\gamma', '\\geq', '\\gt', '\\in', '\\infty', '\\int', '\\lambda', '\\ldots',
     '\\leq', '\\lim', '\\log', '\\lt', '\\mu', '\\neq', '\\phi', '\\pi', '\\pm', '\\prime', '\\rightarrow', '\\sigma',
     '\\sin', '\\sqrt', '\\sum', '\\tan', '\\theta', '\\times', '\\{', '\\}', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|']


    for symbol in keys:
        symbol_dir = path + symbol
        if not os.path.exists(symbol_dir):
            os.makedirs(symbol_dir)
    '''

    #TODO:

    symbols_seen = set()
    path = '/Users/norahborus/Documents/DATA/CLASSES/32x32_Test/'
    from_dir = '/Users/norahborus/Documents/DATA/COMBINED/32x32_All/'
    capital_alphabet = {'A', 'B', 'C', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Y'}
    image_files = os.listdir(from_dir)
    for index,filename in enumerate(image_files):
        if 'kml' not in filename:
            continue
        if 'png' not in filename:
            symbol = filename[filename.index('kml')+3:]
            if  symbol == '.':
                symbol = 'dot'
        else:
            symbol = filename[filename.index('kml')+3:filename.index('png')-1]
            if  symbol == '.':
                symbol = 'dot'

        if symbol in capital_alphabet:
            symbol = 'capital_{}'.format(symbol)

        symbols_seen.add(symbol)
        corresponding_dir = path + symbol + '/'
        if not os.path.exists(corresponding_dir):
            os.makedirs(corresponding_dir)
        shutil.copy(from_dir + filename, corresponding_dir+filename)
        print "Done with file #{}, symbol --> {}".format(index, symbol)


    print "Symbols seen: {}".format(symbols_seen)


def combine_all_data():
    path = '/Users/norahborus/Documents/DATA/training_data/32x32/'
    training_folders = ["CROHME_Characters/","TrainINKML_Characters/",
                        "2012_part1_Characters/", "2012_part2_Characters/"]

    to_dir = '/Users/norahborus/Documents/DATA/COMBINED/32x32_All/'
    for folder in training_folders[1:2]:
        from_dir = path + folder
        image_files = os.listdir(from_dir)
        for index, filename in enumerate(image_files):
            if 'kml' not in filename:
                continue
            if 'png' not in filename:
                symbol = filename[filename.index('kml') + 3:]
                if symbol == 'forward_slash':
                    symbol = '/'
            else:
                symbol = filename[filename.index('kml') + 3:filename.index('png') - 1]
                if symbol == 'forward_slash':
                    symbol = '/'

            shutil.copy(from_dir + filename, to_dir + 'TrainINKML' + '_' + filename)
            print "Folder: {}, Done with file #{}, symbol --> {}".format(folder,index, symbol)


def get_total_distribution():
    distribution = defaultdict(int)
    combined_dir = '/Users/norahborus/Documents/DATA/COMBINED/200x200_All/'
    image_files = os.listdir(combined_dir)
    for index, filename in enumerate(image_files):
        if 'kml' not in filename:
            continue
        if 'png' not in filename:
            symbol = filename[filename.index('kml') + 3:]
            if symbol == 'forward_slash':
                symbol = '/'
        else:
            symbol = filename[filename.index('kml') + 3:filename.index('png') - 1]
            if symbol == 'forward_slash':
                symbol = '/'


        distribution[symbol] += 1
        print "Done with file #{}".format(index)

    od = OrderedDict(sorted(distribution.items()))
    print "DONE WITH COMPILING FREQ INFORMATION"
    print "Total # of classes: {}".format(len(od.keys()))
    plt.rcParams["figure.figsize"] = [16, 9]
    matplotlib.rcParams.update({'font.size': 6})
    y_pos = np.arange(len(od.keys()))
    plt.bar(y_pos, od.values(), align='center', alpha=0.5, color='green')
    plt.xticks(y_pos, od.keys(),rotation=90)
    plt.savefig('DATA_ANALYSIS/aggregate_distribution.png')

    to_file = 'DATA_ANALYSIS/aggregate_class_distribution_pretty_print.txt'
    with open(to_file, 'w') as fp:
        for symbol in od:
            fp.write("{} --> {}\n".format(symbol, od[symbol]))

    with open('DATA_ANALYSIS/aggregate_class_distribution', 'w') as fp:
        cPickle.dump(od, fp)



def class_distribution_per_folder(path, folders, start, end):


    keys = ['!', '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'B', 'C', 'E',
     'F', 'G', 'H', 'I', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Y', '[', '\\Delta', '\\alpha', '\\beta', '\\cos',
     '\\div', '\\exists', '\\forall', '\\gamma', '\\geq', '\\gt', '\\in', '\\infty', '\\int', '\\lambda', '\\ldots',
     '\\leq', '\\lim', '\\log', '\\lt', '\\mu', '\\neq', '\\phi', '\\pi', '\\pm', '\\prime', '\\rightarrow', '\\sigma',
     '\\sin', '\\sqrt', '\\sum', '\\tan', '\\theta', '\\times', '\\{', '\\}', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|']

   # storage_directory = path + name
    distribution = defaultdict(int)


    hw_objects = read_equations(folders, start, end)
   # if not os.path.exists(storage_directory):
   #     os.makedirs(storage_directory)

    for index, hw in enumerate(hw_objects):
        print "Done with #{}".format(index)
      #  print "Inv Mapping: {}".format(hw.inv_mapping)
        symbols = hw.inv_mapping.values()
      #  print 'Symbols: ', symbols
        freq = Counter(symbols)
        for symbol in freq:
            distribution[symbol] += freq[symbol]

    print "Keys: {}".format(distribution.keys())
    print "BEFORE: # of keys: {}".format(len(distribution.keys()))
    for key in keys:
        distribution[key] += 0

    '''
    with open('DATA_ANALYSIS/TrainINKML/class_distribution', 'rb') as fp:
        crohme = cPickle.load(fp)
    
    for symbol in crohme:
        distribution[symbol] += crohme[symbol]
    '''
    od = OrderedDict(sorted(distribution.items()))
    print "AFTER: # of keys: {}".format(len(od.keys()))
  #  print json.dumps(od,indent=4, separators=(',', ': '))
    plt.rcParams["figure.figsize"] = [16, 9]
    matplotlib.rcParams.update({'font.size': 6})
    y_pos = np.arange(len(od.keys()))
    plt.bar(y_pos, od.values(), align='center', alpha=0.5, color='green')
    plt.xticks(y_pos, od.keys(),rotation=90)
    plt.savefig('DATA_ANALYSIS/TrainINKML/distribution.png')

    with open('DATA_ANALYSIS/TrainINKML/class_distribution', 'w') as fp:
        cPickle.dump(od, fp)



def class_distribution_all_folders():
    aggregate_freq = defaultdict(int)
    with open('DATA_ANALYSIS/CROHME/class_distribution', 'rb') as fp:
        crohme = cPickle.load(fp)

    print "Done with crohme"
    with open('DATA_ANALYSIS/TrainINKML/class_distribution', 'rb') as fp:
        train_inkml = cPickle.load(fp)

    print "Done with TrainINKML"
    with open('DATA_ANALYSIS/2012_part1/class_distribution', 'rb') as fp:
        part_1_2012 = cPickle.load(fp)

    print "Done with 2012 part 1"

    with open('DATA_ANALYSIS/2012_part2/class_distribution', 'rb') as fp:
        part_2_2012 = cPickle.load(fp)

    print 'Done with 2012 part 2'
    maps = [crohme, train_inkml, part_1_2012, part_2_2012]
    for map in maps:
        for symbol in map:
            aggregate_freq[symbol] += map[symbol]

    od = OrderedDict(sorted(aggregate_freq.items()))
    print od.keys()
    plt.rcParams["figure.figsize"] = [16, 9]
   # print json.dumps(od,indent=4, separators=(',', ': '))
    matplotlib.rcParams.update({'font.size': 6})
    y_pos = np.arange(len(od.keys()))
    plt.bar(y_pos, od.values(), align='center', alpha=0.5, color='green')
    plt.xticks(y_pos, od.keys(),rotation=90)
    plt.savefig('DATA_ANALYSIS/aggregate_distribution.png')

    with open('DATA_ANALYSIS/aggregate_class_distribution', 'w') as fp:
        cPickle.dump(od, fp)

def random_sample():
    from_dir = '/Users/norahborus/Documents/DATA/CLASSES/32x32_Test/'
    to_dir = '/Users/norahborus/Documents/DATA/CLASSES/random_sample_400/'
    size = 500

    symbols = {'!', '(', ')', '+', ',', '-', 'dot', 'forward_slash', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'capital_A', 'capital_B', 'capital_C', 'capital_E',
     'capital_F', 'capital_G', 'capital_H', 'capital_I', 'capital_L', 'capital_M', 'capital_N', 'capital_P', 'capital_R', 'capital_S', 'capital_T', 'capital_V', 'capital_X', 'capital_Y', '[', '\\Delta', '\\alpha', '\\beta', '\\cos',
     '\\div', '\\exists', '\\forall', '\\gamma', '\\geq', '\\gt', '\\in', '\\infty', '\\int', '\\lambda', '\\ldots',
     '\\leq', '\\lim', '\\log', '\\lt', '\\mu', '\\neq', '\\phi', '\\pi', '\\pm', '\\prime', '\\rightarrow', '\\sigma',
     '\\sin', '\\sqrt', '\\sum', '\\tan', '\\theta', '\\times', '\\{', '\\}', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|'}
    directories = os.listdir(from_dir)
    for dir in directories:
        if dir in symbols:
            filenames = np.array(os.listdir(from_dir + dir + '/'))
            np.random.shuffle(filenames)
            filenames = filenames[:size]
            for filename in filenames:
                shutil.copy(from_dir + dir + '/' + filename, to_dir + filename)


def separate_capital():
    classes_dir = '/Users/norahborus/Documents/DATA/training_data/CLASSES/32x32_Test/'



if __name__ == '__main__':
    path = '/Users/norahborus/Documents/DATA/training_data/'
    training_folders = ['/Users/norahborus/Documents/DATA/training_data/', "CROHME_training_2011/",
                        "TrainINKML_2013/", "trainData_2012_part1/", "trainData_2012_part2/", "MatricesTrain2014/"]
   # class_distribution_per_folder(path, training_folders, 2,3)
   # class_distribution_all_folders()
   # separate_images_by_class()
    random_sample()
 #   create_class_distribution()
   # create_augment_directories()
   # combine_all_data()
  #  get_total_distribution()