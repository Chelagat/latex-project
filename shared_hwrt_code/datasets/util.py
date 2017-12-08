from skimage import data, color, exposure
from skimage.feature import hog

commonly_missegmented_symbols = {'x', '\\times',  'v', '=', '\leq', '\geq'}

def plot_traces(strokes, filename_str):
    """Show the data graphically in a new pop-up window."""

    # prevent the following error:
    # '_tkinter.TclError: no display name and no $DISPLAY environment
    #    variable'
    # import matplotlib
    # matplotlib.use('GTK3Agg', warn=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams['axes.facecolor'] = 'red'
    x, y = [], []

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for stroke_index in range(len(strokes)):
        stroke = strokes[stroke_index]
        xs, ys = [], []
        for p in stroke:
            xs.append(p['x'])
            ys.append(p['y'])
            ax.plot(xs, ys, color="#000000")

    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    plt.gca().invert_yaxis()
    ax.set_aspect('equal')
    plt.axis('off')
    fig.set_size_inches([62.0 / 192, 62.0 / 192])
    fig.canvas.draw()
    filename = "intersection/{}.png".format(filename_str)
    fig.savefig(filename, format='png', facecolor=ax.get_facecolor())
    return filename


def plot_traces_v2(strokes):
    """Show the data graphically in a new pop-up window."""

    # prevent the following error:
    # '_tkinter.TclError: no display name and no $DISPLAY environment
    #    variable'
    # import matplotlib
    # matplotlib.use('GTK3Agg', warn=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams['axes.facecolor'] = 'red'
    x, y = [], []

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for stroke_index in range(len(strokes)):
        stroke = strokes[stroke_index]
        xs, ys = [], []
        for p in stroke:
            xs.append(p['x'])
            ys.append(p['y'])
            ax.plot(xs, ys, color="#000000")

    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    plt.gca().invert_yaxis()
    ax.set_aspect('equal')
    plt.axis('off')

    fig.canvas.draw()
        # fig.savefig("results/"+symbol_str+".png",facecolor=ax.get_facecolor())
        #  np.set_printoptions(threshold=np.nan)
        # Now we can save it to a numpy array.
        #   non_grey = []
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # print data
        # print fig.canvas.get_width_height()
        # print data.shape
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    #  data = np.dot(data[..., :3], [0.299, 0.587, 0.114])
    # print data.shape
    '''
    data_dict = defaultdict(int)
    new_data = np.zeros(data.shape)
    flipped_data = np.zeros(data.shape)
   # print data
    for row in range(len(data)):
        for col in range(len(data[0])):
            point = data[row][col]
            if point != 255:
                flipped_data[row][col] = 0
            else:
                flipped_data[row][col] = 1

            if point != 255:
                new_data[row][col] = 1
                data_dict[(row, col)] = 1

    '''

    #  blurred = gaussian_filter(new_data, sigma=7)
    # im = Image.fromarray(blurred * 255)
    # im.show()
    # print new_data.shape
    # im = Image.fromarray(new_data*255)
    # im.show()
    # new_data = new_data[:, 100:-100]
    #  flipped_data = flipped_data[:,100:-100]
    #  plt.savefig("results/"+symbol_str)
    image = color.rgb2gray(data)

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(32, 32),
                            cells_per_block=(1, 1), visualise=True)

        # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    hog_np_array = np.asarray(hog_image_rescaled)
    return hog_np_array


import random
from sklearn import datasets, svm, metrics
from collections import defaultdict

def svm_linear_test(clf, TEST_X, TEST_Y, y_map):
   # clf.decision_function_shape = "ovr"
    error = 0
    correct = 0
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

def svm_linear_train(TRAINING_X, TRAINING_Y, TEST_Y):
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

    # print len(X), len(Y)
    # for example, y in zip(X, Y):
    #     print len(example), y
    print "About to start fitting"
    print len(TRAINING_X)

    clf = svm.SVC(decision_function_shape='ovo', gamma=0.01, C=100.0)
    clf.fit(TRAINING_X, TRAINING_Y, weights)
    clf.decision_function_shape = "ovr"

    '''
    clf = svm.LinearSVC(multi_class='ovr', C=50.0)
    clf.fit(TRAINING_X, TRAINING_Y, weights)
    '''
    return clf

import cPickle
import numpy as np
from scipy.sparse import csr_matrix
import util

def load_for_segmentation(equation_file):
    X = []
    Y = []

    with open(equation_file, 'rb') as fp:
        rep = cPickle.load(fp)

    symbols = rep['symbol']
    for symbol in symbols:
        symbol_str = symbol[0]
        Y.append(symbol[0])
        np_array_map = symbol[1]
        rows,cols = rep['num_rows'], rep['num_cols']
        np_array = np.zeros((rows,cols))
        for row in xrange(rows):
            for col in xrange(cols):
                if str((row,col)) in np_array_map:
                    np_array[row][col] = np_array_map[str((row,col))]


        sparse_matrix = csr_matrix(np_array)
        X.append(sparse_matrix)

    #print "Done with read"
    return X, Y


import os
import gc
import datetime

def read_files_segmentation(first_path, path):
    filenames = os.listdir(path)
    filenames = [path + file for file in filenames]
    start = datetime.datetime.now()
    TRAINING_X = []
    TRAINING_Y = []
    gc.disable()
    for index, file in enumerate(filenames):
        print "Done --> {}".format(index)
        file = path + file
        x, y = load_for_segmentation(file)
        if x == []:
            continue
        TRAINING_X += x
        TRAINING_Y += y

    gc.enable()

    print len(TRAINING_Y), len(TRAINING_Y)
    TRAINING_X = [(val.toarray()).flatten() for i, val in enumerate(TRAINING_X)]
    end = datetime.datetime.now()
    time_taken = (end - start).total_seconds()
    print "TOTAL TIME TAKEN: {}".format(time_taken)
    return TRAINING_X, TRAINING_Y
