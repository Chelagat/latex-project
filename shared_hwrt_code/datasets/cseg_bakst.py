import sys, os, random
from inkml import read
from ast import literal_eval
import numpy as np
from numpy import array_equal


def f1(t1, t2):
    diff = abs(t1[1] - t2[1])
    if diff == 0:
        return 1
    else:
        return min(1, 1. / diff)


def f2(t1, t2):
    diff = abs(t1[0] - t2[0])
    if diff == 0:
        return 1
    else:
        return min(1, 1. / diff)


def f3(t1, t2):
    return 1 if t1[0] >= t2[1] else 0


def f4(t1, t2):
    return 1 if t1[1] <= t2[0] else 0


def f5(t1, t2):
    diff = abs(t1[2] - t2[2])
    if diff == 0:
        return 1
    else:
        return min(1, 1. / diff)


def f6(t1, t2):
    diff = abs(t1[3] - t2[3])
    if diff == 0:
        return 1
    else:
        return min(1, 1. / diff)


def f7(t1, t2):
    return 1 if t1[0] > t2[0] and t1[1] < t2[1] else 0


def f8(t1, t2):
    return 1. / (((t1[4][0] - t2[4][0]) ** 2 + (t1[4][1] - t2[4][1]) ** 2) ** 0.5)


def featurize(t1, t2):
    return [f1(t1, t2), f2(t1, t2), f3(t1, t2), f4(t1, t2), f5(t1, t2), f6(t1, t2), f7(t1, t2), f8(t1, t2)]


def together(i, j, seg):
    for s in seg:
        if i in s and j in s: return True
    return False


def gather_train_data(_input, _output, _files, directory, parent):
    for index, file in enumerate(_files):
        print "File #{}".format(index)
        hw = read(directory, directory + file, file, parent)
        traces = literal_eval(hw.raw_data_json)

        # Find Trace Bounds:
        trace_bounds = []
        for t in traces:
            max_x, min_x = float('-inf'), float('+inf')
            max_y, min_y = float('-inf'), float('+inf')
            avg = [0.0, 0.0]
            for d in t:
                y = d['y']
                x = d['x']
                avg[0] += x
                avg[1] += y
                if y > max_y:
                    max_y = y
                elif y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                elif x < min_x:
                    min_x = x
            avg[0] = avg[0] / len(t)
            avg[1] = avg[1] / len(t)
            trace_bounds.append((max_x, min_x, max_y, min_y, avg))

        # Create feature vector
        for i, t1 in enumerate(trace_bounds):
            for j, t2 in enumerate(trace_bounds):
                if i == j: continue
                feature = featurize(t1, t2)
                _input.append(feature)
                _output.append(together(i, j, hw.segmentation))


def create_tests(_input, _output, _files, directory, parent):
    for index, file in enumerate(_files):
        print "CREATE TESTS: file #{}".format(index)
        hw = read(directory, directory + file, file, parent)
        traces = literal_eval(hw.raw_data_json)

        # Find Trace Bounds:
        trace_bounds = []
        for t in traces:
            max_x, min_x = float('-inf'), float('+inf')
            max_y, min_y = float('-inf'), float('+inf')
            avg = [0.0, 0.0]
            for d in t:
                y = d['y']
                x = d['x']
                avg[0] += x
                avg[1] += y
                if y > max_y:
                    max_y = y
                elif y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                elif x < min_x:
                    min_x = x
            avg[0] = avg[0] / len(t)
            avg[1] = avg[1] / len(t)
            trace_bounds.append((max_x, min_x, max_y, min_y, avg))
        _input.append(trace_bounds)
        _output.append(hw.segmentation)


def test(clf, test_input, test_output):
    total_input, correct_input, total_chars, correct_chars = len(test_input), 0, 0, 0
    for i, tb in enumerate(test_input):

        check = range(0, len(tb))
        groupings = []
        current_group = []
        while len(check) > 0:
            remaining = []
            current_group.append(check[0])
            for j in range(1, len(check)):
                ### CLASSIFY HERE###
                input = featurize(tb[check[0]],tb[check[j]])
              #  print "Input: {}".format(input)
                same = clf.predict([input])[0]
                ####################
                if not same:
                    remaining.append(check[j])
                else:
                    current_group.append(check[j])
            groupings.append(current_group)
            current_group = []
            check = remaining[:]

        # Gather Accuracy Data For Each Character
        total_chars += len(test_output[i])
        for g in groupings:
            if g in test_output[i]: correct_chars += 1

        if np.array_equal(groupings, test_output[i]): correct_input += 1

    return total_input, correct_input, total_chars, correct_chars


def split_data(files):
    random.seed(123)
    test_files = []
    for i in range(len(files) / 10):
        test_files.append(random.choice(files))
        files.remove(test_files[i])
    return files, test_files


from sklearn import svm
from sklearn import linear_model

def main():
    directories = ['/Users/norahborus/Documents/DATA/training_data/CROHME_training_2011/']
    parents = ['CROHME_training_2011']

    training_input, training_output = [], []
    test_input, test_output = [], []

    for i in range(len(directories)):
        files = os.listdir(directories[i])
        training_files, test_files = split_data(files)
        print "DONE WITH SPLIT"
        gather_train_data(training_input, training_output, training_files, directories[i], parents[i])
        print training_input[0], training_output[0]
        print "DONE WITH GATHER"
        create_tests(test_input, test_output, test_files, directories[i], parents[i])
        print "Test input: ", test_input[0], "Test output: ", test_output[0]
        print "DONE WITH CREATE TESTS"
        print len(training_input), training_input[0], training_output[0]

    #### TRAIN SVM HERE ####
    print "TRAINING SVM"
    clf = svm.LinearSVC(C=50)
    clf.fit(training_input, training_output)
    ########################

    print "DONE TRAINING SVM"
    total_input, correct_input, total_chars, correct_chars = test(clf, test_input, test_output)

    print 'ACCURACY: ', correct_chars, '/', total_chars, ' = ', float(correct_chars)/total_chars
    print 'EQUATION ACCURACY: ', correct_input, '/', 'total_input', ' = ', float(correct_input)/total_input


if __name__ == '__main__':
    main()