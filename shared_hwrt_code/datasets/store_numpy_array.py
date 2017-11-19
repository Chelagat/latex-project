from inkml import read_equations
import pickle
import os
import json
from ast import literal_eval
import numpy as np
import itertools
import matplotlib.pyplot as plt
from inkml import svm_train

def store_info(path, folders, start,end):
    '''
    Update: stored 100 equations so far in crohme, start running from file 101
    '''
    #Idea: store sparse dictionary
    hw_objects = read_equations(folders, start, end)
    storage_directory = path + str(start) + str(end) + '/'

    if not os.path.exists(storage_directory):
        os.makedirs(storage_directory)

    for index, hw in enumerate(hw_objects):
        print "Done with #{}".format(index)
        x,y = hw.get_training_example()
        rep = {}

        for np_array, symbol in zip(x,y):
            #Convert sparse np array to dictionary
          #  print symbol
            rows, cols =  np_array.shape
            rep['num_rows'] = rows
            rep['num_cols'] = cols
            np_array_map = {}
            for row in range(rows):
                for col in range(cols):
                    if np_array[row][col] != 0:
                        np_array_map[str((row,col))] = np_array[row][col]

            if 'symbol' not in rep:
                rep['symbol'] = [[symbol, np_array_map]]
            else:
                rep['symbol'].append([symbol, np_array_map])


        filename = storage_directory + hw.filename
        with open(filename, "w") as fp:
            json.dump(rep, fp)

    '''
    storage_directory_svm = storage_directory + "svm/"
    if not os.path.exists(storage_directory_svm):
        os.makedirs(storage_directory_svm)

    filename = storage_directory_svm + "svm.pickle"
    with open(filename, "w") as fp:
        pickle.dump(svm, fp)
    
    '''


def load_info(path, folder):
    print "***********************************************"
    storage_directory = path + folder
    if not os.path.exists(storage_directory):
        raise ValueError("Directory to load training data from does not exist!")

    TRAINING_X = []
    TRAINING_Y = []
    equation_files = os.listdir(storage_directory)
    for equation_file in equation_files:
        equation_file = storage_directory + equation_file
        with open(equation_file, 'rb') as fp:
            data = fp.read()
            rep = json.loads(data)

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


    return TRAINING_X, TRAINING_Y

def main():

    path = '/Users/norahborus/Documents/latex-project/baseline/training_data/'
    training_folders = ["/Users/norahborus/Documents/latex-project/baseline/training_data/", "CROHME_training_2011/", "TrainINKML_2013/", "trainData_2012_part1/", "trainData_2012_part2/"]
   # store_info(path, training_folders, 1,2)
  #  print "STORED EVERYTHING FOR FOLDER: {}!".format(training_folders[4:5])
    X, Y = load_info(path, "CROHME_training_2011_JSON/")
    print "Training and svm predicting for data in: CROHME_training_2011_JSON/"
    svm_train(X,Y)



if __name__ == '__main__':
    main()
