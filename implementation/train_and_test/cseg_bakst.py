import sys, os, random
from inkml import read
from ast import literal_eval
import numpy as np
from numpy import array_equal
import pickle, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# Horizontal containment
def f7(t1, t2):
    return 1 if (t1[0] > t2[0] and t1[1] < t2[1]) else 0

# Distance between center of mass (avarge points of 2 traces)
def f8(t1, t2):
    return 1. / (((t1[4][0] - t2[4][0]) ** 2 + (t1[4][1] - t2[4][1]) ** 2) ** 0.5)

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def GetAngleOfLineBetweenTwoPoints(p1, p2):
        xDiff = p2[0] - p1[0]
        yDiff = p2[1] - p1[1]
        return math.degrees(math.atan2(yDiff, xDiff))

def f9(t1, t2):
    angle_between_points = GetAngleOfLineBetweenTwoPoints(t1[4], t2[4])
    if angle_between_points > 180:
        angle_between_points = angle_between_points - 180
    if angle_between_points < 45 or angle_between_points > 135:
        return 1
    else:
        return 0



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


def create_tests(_input, _output, _traces, _files, directory, parent):
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
        _traces.append(traces)

# Given a list of traces (a list of lists of points),
# plots all the traces in the list on the figure named
# based on the index passed in.
def plot_traces(traces, index, ax):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for trace in traces:
        xs, ys = [], []
        for p in trace:
            xs.append(p['x'])
            ys.append(p['y'])
            ax.plot(xs,ys, color="#000000")
    plt.gca().invert_yaxis()
    ax.set_aspect('equal')
    plt.axis('off')

    fig.canvas.draw()
    fig.savefig("results/"+ "incorrect_seg" +str(index)+".png",facecolor=ax.get_facecolor())

# Given the parameters, tests the error (calculates accuracy and plots incorrectly group traces)
# clf - trained SVM classifier
def test(clf, test_input, test_output, traces, plot_incorrects):
    print "len test input: ", len(test_input), "len_traces: ", len(traces)
    total_input, correct_input, total_chars, correct_chars = len(test_input), 0, 0, 0
    correct_long, incorrect_short = 0, 0
    _, ax = plt.subplots()
    incorrect_ind = 0
    for i, tb in enumerate(test_input):
        check = range(0, len(tb))
        groupings, traces_groupings = [], []
        current_group, current_group_traces = [], []
        curr_char_traces = traces[i]
        while len(check) > 0:
            remaining, remaining_traces = [], []
            current_group.append(check[0])
            current_group_traces.append(curr_char_traces[0])
            for j in range(1, len(check)):
                ### CLASSIFY HERE###
                input = featurize(tb[check[0]],tb[check[j]])
              #  print "Input: {}".format(input)
                same = clf.predict([input])[0]
                ####################
                if not same:
                    remaining.append(check[j])
                    remaining_traces.append(curr_char_traces[j])
                else:
                    current_group.append(check[j])
                    current_group_traces.append(curr_char_traces[j])                                                                                                                  
            groupings.append(current_group)
            traces_groupings.append(current_group_traces)
            current_group, current_group_traces = [], []
            check = remaining[:]
            curr_char_traces = remaining_traces[:]

        # Gather Accuracy Data For Each Character
        total_chars += len(test_output[i])
        sorted_test_output_i = [sorted(elem) for elem in test_output[i]]
        sorted_groupings = [sorted(elem) for elem in groupings]

        for j, g in enumerate(sorted_groupings):
            if g in sorted_test_output_i:
                correct_chars += 1
            else:
                print "Incorrect character: ", g, "correct grouping: ", sorted_test_output_i, "index: ", incorrect_ind
                if plot_incorrects: plot_traces(traces_groupings[j], incorrect_ind, ax)
                incorrect_ind += 1

        if np.array_equal(groupings, test_output[i]): correct_input += 1
    print "correct long: ", correct_long, "incorrect short: ", incorrect_short
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

def train_accuracy(clf, input, output):
    total_chars, total_correct = 0, 0
    for i, elem in enumerate(input):
        if clf.predict([elem])[0] == output[i]:
            total_correct += 1
        total_chars += 1
    return total_chars, total_correct

def main():
    train_from_scrach, plot_incorrects = True, False
    directories = ['/Users/amit/latex-project/shared_hwrt_code/datasets/training_data/CHROME_training_2011/']
    parents = ['CROHME_training_2011']
    train_accuracy_list, test_accuracy_list = [], []
    training_input, training_output = [], []
    test_input, test_output, traces = [], [], []
    clf = None
    c_vals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    # Read files and train the SVM
    if True:
        for i in range(len(directories)):
            files = os.listdir(directories[i])
            training_files, test_files = split_data(files)
            print "DONE WITH SPLIT"
            gather_train_data(training_input, training_output, training_files, directories[i], parents[i])
            print training_input[0], training_output[0]
            print "DONE WITH GATHER"
            create_tests(test_input, test_output, traces, test_files, directories[i], parents[i])
            print "Test input: ", test_input[0], "Test output: ", test_output[0]
            print "DONE WITH CREATE TESTS"
            print len(training_input), training_input[0], training_output[0]
    
    for c_val in c_vals:
        if train_from_scrach:
            #### TRAIN SVM HERE ####
            print "TRAINING SVM"
            clf = svm.LinearSVC(C=c_val)
            clf.fit(training_input, training_output)
            with open('clf.pkl', 'w') as f:  # Python 3: open(..., 'wb')
                pickle.dump([clf, test_input, test_output, traces], f)
            ########################
        # Get trained SVM from file
        else:
            with open('clf.pkl') as f:  # Python 3: open(..., 'rb')
                clf, test_input, test_output, traces = pickle.load(f)
        print "DONE TRAINING SVM"
        total_input, correct_input, total_chars, correct_chars = test(clf, test_input, test_output, traces, plot_incorrects)

        # Test accuracy
        print ' Test ACCURACY: ', correct_chars, '/', total_chars, ' = ', float(correct_chars)/total_chars
        print 'EQUATION ACCURACY: ', correct_input, '/', 'total_input', ' = ', float(correct_input)/total_input
        test_accuracy_list.append(float(correct_chars)/total_chars)
        
        # Train accuracy
        total_chars, total_correct = train_accuracy(clf, training_input, training_output)
        print 'Train ACCURACY: ', total_correct, '/', total_chars, ' = ', float(total_correct)/total_chars
        # print 'EQUATION ACCURACY: ', correct_input, '/', 'total_input', ' = ', float(correct_input)/total_input
        train_accuracy_list.append(float(total_correct)/total_chars)

    with open('graph.pkl', 'w') as f:  # Python 3: open(..., 'wb')
        pickle.dump([train_accuracy_list, test_accuracy_list], f)

    train_graph, = plt.plot(c_vals, train_accuracy_list, label='Train Accuracy', color='red')
    test_graph, = plt.plot(c_vals, test_accuracy_list, label='Test Accuracy', color='green')
    plt.ylabel('Accuracy')
    plt.xlabel('Regularization constant')
    plt.legend([train_graph, test_graph], ['Train Accuracy', 'Test Accuracy'])
    plt.savefig('SVM_CSeg_Cval_graph.png')

if __name__ == '__main__':
    main()