import os
from parallelism_read import read_images_flattened
from util import *
from data_augmentation import read_images_svm_augmented
import cPickle
from PIL import Image


def train_svm():
    path = '/Users/norahborus/Documents/DATA/training_data/'
  #  folders = ['/Users/norahborus/Documents/DATA/COMBINED/', '200x200_All/', '200x200_All/']
    folders = ['/Users/norahborus/Documents/DATA/training_data/HOG_200x200/', 'CROHME_images_quarter_size/','CROHME_images_quarter_size/']
  #  folders = ['/Users/norahborus/Documents/DATA/training_data/', 'CROHME_images_half_size/','CROHME_images_half_size/']
    TRAINING_X, TRAINING_Y, TEST_X, TEST_Y, y_map = read_images_flattened(folders)
    print len(TRAINING_Y), len(TEST_Y)
    clfs, y_map = svm_linear_train(TRAINING_X, TRAINING_Y, TEST_Y)
    svm_linear_test(clfs[0], TEST_X, TEST_Y, TRAINING_X, TRAINING_Y, y_map)
    dev_history = []
    train_history = []
    '''
    for clf in clfs:
        dev_accuracy, train_accuracy = svm_linear_test(clf, TEST_X, TEST_Y, TRAINING_X, TRAINING_Y, y_map)
        dev_history.append(dev_accuracy)
        train_history.append(train_accuracy)

   # x_labels = [0.001,0.002, 0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
    x_labels = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    plt.plot(x_labels, train_history, color = 'red', label = 'Train Accuracy')
    plt.plot(x_labels, dev_history, color='green', label='Dev Accuracy')
    plt.legend(loc='best')
    plt.ylabel('Accuracy')
    plt.xlabel('C Value')
    plt.savefig("DATA_ANALYSIS/linear_c_value_tuning_hog_1000.png")
    '''

def store_data_for_codalab():
  #  folders = ['/Users/norahborus/Documents/DATA/COMBINED/', '200x200_All/', '200x200_All/']
    folders = ['/Users/norahborus/Documents/DATA/training_data/HOG_200x200/', 'CROHME_images_quarter_size/','CROHME_images_quarter_size/']
    #  folders = ['/Users/norahborus/Documents/DATA/training_data/', 'CROHME_images_half_size/','CROHME_images_half_size/']
    TRAINING_X, TRAINING_Y, TEST_X, TEST_Y, y_map = read_images_flattened(folders)

    
    with open('train_x', 'w') as fp:
        cPickle.dump(TRAINING_X, fp)

    print "Done with train_x"
    with open('train_y', 'w') as fp:
        cPickle.dump(TRAINING_Y, fp)

    print "Done with train_y"
    with open('test_x', 'w') as fp:
        cPickle.dump(TEST_X, fp)

    print "Done with test_x"
    with open('test_y', 'w') as fp:
        cPickle.dump(TEST_Y, fp)

    print "Done with test_"

def load_data_for_codalab():
    with open('y_map', 'rb') as fp:
        y_map = cPickle.load(fp)

    with open('train_x', 'rb') as fp:
        TRAINING_X = cPickle.load(fp)

    with open('train_y', 'rb') as fp:
        TRAINING_Y = cPickle.load(fp)

    with open('test_x', 'rb') as fp:
        TEST_X = cPickle.load(fp)

    with open('test_y', 'rb') as fp:
        TEST_Y = cPickle.load(fp)

    clfs, y_map = svm_linear_train(TRAINING_X, TRAINING_Y, TEST_Y)
    svm_linear_test(clfs[0], TEST_X, TEST_Y, TRAINING_X, TRAINING_Y, y_map)

def load_info_v3(equation_file, downsize=False):
   # print "***********************************************"
    print "START: {}".format(equation_file)
    TRAINING_X = []
    TRAINING_Y = []


    with open(equation_file, 'rb') as fp:
        rep = cPickle.load(fp)

    symbols = rep['symbol']
   # print "Symbols: {}".format(symbols)
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

        if symbol[0] == '\phi':
            print "HERE"
            return np_array
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

import matplotlib.pyplot as plt
def compare_outputs():
    file_image = "/Users/norahborus/Documents/DATA/training_data/CROHME_images_half_size/_formulaire001-equation001.inkml\phi.png"
    image = Image.open(file_image).convert('L')

    array_1 = np.asarray(image)
    equation_file = "/Users/norahborus/Documents/DATA/training_data/CROHME_half_size/formulaire001-equation001.inkml"
    array_2 = load_info_v3(equation_file)

    array_1  = array_1 / 255.0
    plt.imshow(array_1)
    plt.show()
    plt.savefig("array_1")
    print array_1
    print array_2
    plt.imshow(array_2)
    plt.show()
    plt.savefig("array_2")
    print "array 1 shape: ", array_1.shape
    print "array 2 shape: ", array_2.shape

    for row in range(len(array_1)):
        print "*****************************"
        for val_1, val_2 in zip(array_1[row], array_2[row]):
            print val_1 , "," ,val_2 ,

        print " "
        print "*****************************"
    print "Are the 2 arrays the same?"
    print np.max(array_1), np.max(array_2)

if __name__ == '__main__':
  #  train_svm()
    store_data_for_codalab()
   # load_data_for_codalab()
  #  compare_outputs()