import os
from parallelism_read import read_images_flattened
from util import *
from sklearn.ensemble import RandomForestClassifier


def rf_test(clf, TEST_X, TEST_Y,TRAINING_X, TRAINING_Y, y_map):
   # clf.decision_function_shape = "ovr"
    error = 0
    correct = 0
    class_total = defaultdict(int)
    class_correct = defaultdict(int)
    '''
    for i in range(len(TRAINING_X)):
        max_index = clf.predict([TRAINING_X[i]])[0]
        #   print "Len dec: {}, Dec: {}".format(len(dec[0]), dec[0])
        # print "Max: {}".format(max(dec[0]))
        #  print "Max index: {}".format(max_index)
        for symbol, index in y_map.iteritems():
            if index == max_index:
                #  print "Matching symbol: {}, Truth: {}".format(symbol, TEST_Y[i])
                if symbol != TRAINING_Y[i]:
                    error += 1
                else:
                    class_correct[symbol] += 1
                    correct += 1

                class_total[symbol] += 1

    print "TRAINING Accuracy: {}".format(100.0 * correct / len(TRAINING_Y))
    print "TRAINING Error: {}".format(100.0 * error / len(TRAINING_Y))

    train_accuracy = 100.0 * correct / len(TRAINING_Y)

    print "CLASS ACCURACY:"
    for symbol in y_map:
        if class_total[symbol] == 0:
            continue
        accuracy = 100.0 * class_correct[symbol] / class_total[symbol]
        print "Class --> {}, Accuracy --> {}".format(symbol, accuracy)
    '''

    error = 0
    correct = 0
    class_total = defaultdict(int)
    class_correct = defaultdict(int)
    for i in range(len(TEST_X)):
        max_index = clf.predict([TEST_X[i]])[0]
        #   print "Len dec: {}, Dec: {}".format(len(dec[0]), dec[0])
        # print "Max: {}".format(max(dec[0]))
        #  print "Max index: {}".format(max_index)
        for symbol, index in y_map.iteritems():
            if index == max_index:
                #  print "Matching symbol: {}, Truth: {}".format(symbol, TEST_Y[i])
                if symbol != TEST_Y[i]:
                    error += 1
                else:
                    class_correct[symbol] += 1
                    correct += 1

                class_total[symbol] += 1


    #print "TEST Accuracy: {}".format(100.0 * correct / len(TEST_Y))
    #print "TEST Error: {}".format(100.0 * error / len(TEST_Y))

    test_accuracy = 100.0 * correct / len(TEST_Y)

    '''
    print "CLASS ACCURACY:"
    for symbol in y_map:
        if class_total[symbol] == 0:
            continue
        accuracy = 100.0 * class_correct[symbol] / class_total[symbol]
        print "Class --> {}, Accuracy --> {}".format(symbol, accuracy)
    '''

    print "********************************************************"


    print "********************************************************"
    print "*****************SUMMARY******************************"

   # print "TRAIN Accuracy: {}".format(train_accuracy)
    print "TEST Accuracy: {}".format(test_accuracy)

    return test_accuracy

def rf_train(TRAINING_X, TRAINING_Y, TEST_Y):
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

    Y = NEW_TRAINING_Y

    for y in TEST_Y:
        if y in y_map:
            continue
        y_map[y] = counter
        counter += 1

    print "About to start fitting"
    print len(TRAINING_X)

    max_features = range(5,50,5)
    n_estimators = range(50,550, 100)
    combined = {}
    for val_2 in n_estimators :
        curr_list = []
        for val_1 in max_features:
            clf = RandomForestClassifier(max_depth=10, random_state=0, max_features=val_1, n_estimators=val_2)
            clf.fit(TRAINING_X, Y, weights)
            curr_list.append(clf)

        print "Done with outer"
        combined[val_2] = curr_list

    return combined, y_map

import matplotlib.pyplot as plt

def load_rf():
    color_array = [
        "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941",
        "#006FA6", "#A30059", "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC",
        "#B79762", "#004D43", "#8FB0FF", "#997D87", "#5A0007", "#809693",
        "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9",
        "#B903AA", "#D16100", "#DDEFFF", "#000035", "#7B4F4B", "#A1C299",
        "#300018", "#0AA6D8", "#013349", "#00846F", "#372101", "#FFB500",
        "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68",
        "#7A87A1", "#788D66", "#885578", "#FAD09F", "#FF8A9A", "#D157A0",
        "#BEC459", "#456648", "#0086ED", "#886F4C",

        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9",
        "#FF913F", "#938A81", "#575329", "#00FECF", "#B05B6F", "#8CD0FF",
        "#3B9700", "#04F757", "#C8A1A1", "#1E6E00", "#7900D7", "#A77500",
        "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0",
        "#3A2465", "#922329", "#5B4534", "#FDE8DC", "#404E55", "#0089A3",
        "#CB7E98", "#A4E804", "#324E72", "#6A3A4C", "#83AB58", "#001C1E",
        "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
        "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4",
        "#1E0200", "#5B4E51", "#C895C5", "#320033", "#FF6832", "#66E1D3",
        "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"]

    path = '/Users/norahborus/Documents/DATA/training_data/'
  #  folders = ['/Users/norahborus/Documents/DATA/training_data/', 'TrainINKML_images_quarter_size/', 'TrainINKML_images_quarter_size/']
    folders = ['/Users/norahborus/Documents/DATA/training_data/32x32/', 'CROHME_Characters/','CROHME_Characters/']
   # folders = ['/Users/norahborus/Documents/DATA/CLASSES/', 'random_sample_200/', 'random_sample_200/']
    TRAINING_X, TRAINING_Y, TEST_X, TEST_Y, y_map = read_images_flattened(folders)
    print len(TRAINING_Y), len(TEST_Y)
    combined, y_map = rf_train(TRAINING_X, TRAINING_Y, TEST_Y)
    train_accuracy = []
    dev_accuracy = []

    key_index = 0
    for key in combined:
        estimators = combined[key]
        dev_accuracy = []
        for clf in estimators:
            dev = rf_test(clf, TEST_X, TEST_Y, TRAINING_X, TRAINING_Y, y_map)
            dev_accuracy.append(dev)

        plt.plot(range(5,50,5),dev_accuracy ,color=color_array[key_index], label='num_estimators={}'.format(key))

        key_index +=1

    plt.legend(loc='best')
   # plt.savefig('DATA_ANALYSIS/max_features_accuracy.png')

    plt.savefig('DATA_ANALYSIS/rf_param_tuning_part2.png')
if __name__ == '__main__':
    load_rf()