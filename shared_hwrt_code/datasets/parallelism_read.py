import thread
import time
import pickle
import os
import json
from ast import literal_eval
import numpy as np
import itertools
#from inkml import svm_train
import threading
from multiprocessing import Pool
import datetime
import cPickle
from inkml import read_equations


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

        #plt.imshow(np_array,cmap='gray')
        #plt.savefig("results/{}".format(symbol[0]))
        TRAINING_X.append(list(itertools.chain.from_iterable(np_array)))

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

def read_files_sequence(path):

    print path
    filenames = os.listdir(path)
    print filenames
    filenames = [path+file for file in filenames]
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

    for index, file in enumerate(filenames):
        print "Done --> {}".format(index)
        if index % 10 == 0:
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

    end = datetime.datetime.now()
    time_taken = (end-start).total_seconds()
    print "TOTAL TIME TAKEN: {}".format(time_taken)
   # print len(data), type(data[0])
  #  print data[0]

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
   # print len(data), type(data[0])
   # print data[0]



def main():
    path = '/Users/norahborus/Documents/DATA/'
    training_folders = ['/Users/norahborus/Documents/DATA/', "CROHME_training_2011/", "TrainINKML_2013_JSON/", "trainData_2012_part1_JSON/", "trainData_2012_part2_JSON/"]
   # store_info(path, training_folders, 1, 2)
   # print "*************SEQUENCE*****************"
    read_files_sequence(path + "12/")
  #  print "***************************************"
   # print "***************THREADS******************"
  #  read_files_thread(path + "12/")





if __name__ == '__main__':
    main()
