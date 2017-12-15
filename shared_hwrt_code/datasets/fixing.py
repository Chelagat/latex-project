import os
from parallelism_read import read_images_flattened
from util import *




import numpy as np
import cPickle

import matplotlib.pyplot as plt

def nn_accuracy():
    with open("nn_accuracy", "rb") as fp:
        nn_accuracy = cPickle.load(fp)

    with open("cnn_accuracy", "rb") as fp:
        cnn_accuracy = cPickle.load(fp)

    cnn_accuracy, = plt.plot(range(len(cnn_accuracy)), cnn_accuracy, label='CNN', color='red')
    nn_accuracy, = plt.plot(range(len(nn_accuracy)), nn_accuracy , label='NN', color='green')
    plt.legend([cnn_accuracy, nn_accuracy], ['CNN', '1 layer NN'])
    plt.xlabel("Number of Epochs")
    plt.ylabel("Train Accuracy")
    plt.savefig('cnn_vs_nn_accuracy.png')

def split_data_for_poster_session():
   # equation_from_directory = "/Users/norahborus/Documents/DATA/COMBINED/TRAIN_characters/"
    from_directory = "/Users/norahborus/Documents/DATA/training_data/POSTER_SESSION/12_characters/"
    to_directory = "/Users/norahborus/Documents/DATA/training_data/POSTER_SESSION/TEST_characters/"
    to_size = 665
    from_filenames = np.array(os.listdir(from_directory))
    print len(os.listdir(from_directory)), len(os.listdir(to_directory))
    np.random.shuffle(from_filenames)
    for filename_orig in from_filenames[:665]:
        filename = from_directory + filename_orig
        new_filename = to_directory + filename_orig
        os.rename(filename, new_filename)


    print len(os.listdir(from_directory)), len(os.listdir(to_directory))

def move_files():
    from_directory = "/Users/norahborus/Documents/DATA/COMBINED/TRAIN_characters/"
    to_directory = "/Users/norahborus/Documents/DATA/COMBINED/TEST_characters/"
    from_filenames = np.array(os.listdir(from_directory))
    print len(os.listdir(from_directory)), len(os.listdir(to_directory))
    np.random.shuffle(from_filenames)
    print len(from_filenames)
    for filename_orig in from_filenames[:28]:
        filename = from_directory + filename_orig
        new_filename = to_directory + filename_orig
       # print filename, new_filename
        os.rename(filename, new_filename)
   
    print len(os.listdir(from_directory)), len(os.listdir(to_directory))
    '''
    for filename in filenames:
        filename = directory + filename
        if '.png' in filename:
            continue

        if 'png' not in filename:
            os.remove(filename)
            continue

        fixed_name = filename[:filename.index('png')] + '.' + filename[filename.index('png'):]
        os.rename(filename, fixed_name)
    '''

nn_accuracy()