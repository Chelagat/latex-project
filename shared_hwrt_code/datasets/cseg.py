import sys, os, random
from inkml import read
from ast import literal_eval
import numpy as np
from numpy import array_equal

from collections import defaultdict
from operator import itemgetter
import itertools
from sklearn import datasets, svm, metrics

def traceDistance(trace_1, trace_2):
    min_distance = float('inf')
   # print trace_1
  #  print trace_2
    for coord1 in trace_1:
        for coord2 in trace_2:
           # print (coord1.strip()).split(" ")
           # print (coord2.strip()).split(" ")
            x1, y1 = (coord1.strip()).split(" ")
            x2, y2 = (coord2.strip()).split(" ")
            distance = ((float(x1) - float(x2))**2 + (float(y1) - float(y2))**2)**0.5
            min_distance = min(min_distance, distance)

    #print min_distance
    return min_distance

def normalize(Y):
    return 1.0*Y / np.sum(Y)


from util import plot_traces
from PIL import Image
#import nn_with_inkml

INTERSECTION_PROBABILITY_THRESHOLD = 0.7
import util
def min_distance_segment(so_far, hw, nn_params=None, pytorch_params=None, use_pset4_cnn =False, use_pytorch_cnn= False, ):
    import cnn_from_pset4 as cnn
    import nn_with_grayscale as pytorch_cnn
    min_global = 5
    strokes = hw.strokes
    pointlist = hw.get_pointlist()
   # print len(pointlist)
    neighbors = defaultdict(list)
    ocr_used = 0
    INTERSECTION_PROBABILITY_THRESHOLD = 0.7

    if use_pset4_cnn:
        y_map = nn_params['y_map']
        num_classes = nn_params['num_classes']
        inv_map = nn_params['inv_map']
        params = nn_params['params']

    if use_pytorch_cnn:
        y_map  = pytorch_params['y_map']
        num_classes = pytorch_params['num_classes']
        inv_map = pytorch_params['inv_map']


    for i, curr_trace in enumerate(strokes[:-1]):
        min_distance = float('inf')
        for j, other_trace in enumerate(strokes[i + 1:]):
            index = j + i + 1
            distance = traceDistance(curr_trace, other_trace)
            if distance <= min_global:
                neighbors[i].append(index)
            elif use_pset4_cnn and abs(i - index) < 2:
                group = [pointlist[i], pointlist[index]]
                image_file = plot_traces(group, str(so_far) + '_' + str(i) + '_' + str(index))
                array = np.asarray(Image.open(image_file).convert('L'))
                data = [array.flatten()]
                data = np.array(data, dtype=np.float128)
                prediction = cnn.predict(data, params, num_classes)[0]
                max_prediction = inv_map[np.argmax(prediction)]
                top_predictions = np.argsort(prediction)[::-1][:1]
                max_probability = prediction[top_predictions[0]]
                summary = {}
                if max_probability > INTERSECTION_PROBABILITY_THRESHOLD:
                    for symbol in util.commonly_missegmented_symbols:
                        if symbol not in y_map:
                            continue
                        id = y_map[symbol]
                        if id in top_predictions:
                            print "value of i: {}, to_check: {}, OCR for segmentation".format(i, index)
                            ocr_used += 1
                            neighbors[i].append(index)

               # print "************MAX PREDICTION: {} **** INTERSECTION PROBABILITIES: {}".format(max_prediction, summary)
            elif use_pytorch_cnn:
                group = [pointlist[i], pointlist[index]]
                image_file = plot_traces(group, str(so_far) + '_' + str(i) + '_' + str(index))
                image = Image.open(image_file)
                image_tensor = pytorch_cnn.transform(image)
                net, classes = pytorch_cnn.main()
                outputs = pytorch_cnn.test_single_example(image_tensor, net, classes)
                for position, prob in enumerate(outputs):
                    symbol = inv_map[position]
                    if symbol in util.commonly_missegmented_symbols and prob > INTERSECTION_PROBABILITY_THRESHOLD:

                        print "value of i: {}, to_check: {}, OCR for segmentation".format(i, index)
                        ocr_used += 1
                        neighbors[i].append(index)
                        continue

    groups = []
    values = list(itertools.chain.from_iterable(neighbors.values()))
    neighbors_copy = {}
    seen = set()
    for key, vals in neighbors.iteritems():
        if key in seen:
            continue
        neighbors_copy[key] = vals

        for val in vals:
            if val in neighbors:
                neighbors_copy[key] = list(set(neighbors_copy[key] + neighbors[val]))
                seen.add(val)

    neighbors = neighbors_copy
    for k in range(len(pointlist)):
        if k not in neighbors.keys() and k not in values:
            groups.append([k])

    for key, val in neighbors.iteritems():
        group = [key] + val
        groups.append(group)

    groups = sorted(groups, key=itemgetter(0))
    return groups, ocr_used



def segmentation_svm_min_distance(recordings, use_pset4_cnn = False, use_pytorch_cnn = True):
    import cnn_from_pset4 as cnn
    import nn_with_grayscale as pytorch_cnn
    import util
    import cPickle

    if use_pset4_cnn:
        # params, num_classes, y_map = cnn.main()
      #  y_map, num_classes = cnn.main()
      #  with open('cseg_y_map', 'w') as fp:
      #      cPickle.dump(y_map, fp)
      #  with open('cseg_num_classes', 'w') as fp:
      #      cPickle.dump(num_classes, fp)

        with open('cseg_y_map', 'rb') as fp:
            y_map = cPickle.load(fp)
        with open('cseg_num_classes', 'rb') as fp:
            num_classes = cPickle.load(fp)
        params = {}
        with open('regularized_W1', 'rb') as fp:
            params['W1'] = cPickle.load(fp)
        with open('regularized_b1', 'rb') as fp:
            params['b1'] = cPickle.load(fp)
        with open('regularized_W2', 'rb') as fp:
            params['W2'] = cPickle.load(fp)
        with open('regularized_b2', 'rb') as fp:
            params['b2'] = cPickle.load(fp)

        inv_map = {v: k for k, v in y_map.iteritems()}
        nn_params = {}
        nn_params['y_map'] = y_map
        nn_params['inv_map'] = inv_map
        nn_params['num_classes'] = num_classes
        nn_params['params'] = params

    if use_pytorch_cnn:

        net, classes, y_map = pytorch_cnn.main()
        inv_map = {v: k for k, v in y_map.iteritems()}
        pytorch_params = {}
        pytorch_params['net'] = net
        pytorch_params['classes'] = classes
        pytorch_params['y_map'] = y_map
        pytorch_params['inv_map'] = inv_map

    total_correct, total, total_correct_chars, total_chars = 0, 0, 0, 0
    history = []
    ocr_count = 0
    for index, hw in enumerate(recordings):
        print "SEPARATION: File #{}".format(index)
        if hw == None:
            continue

        seg_guess, ocr_used = min_distance_segment(index, hw, nn_params=nn_params, pytorch_params=pytorch_params,
                                         use_pset4_cnn=False, use_pytorch_cnn=True)

        ocr_count += ocr_used

        history.append((seg_guess, hw.segmentation, hw.baseline_parsed))
        for g in seg_guess:
            if g in hw.segmentation: total_correct_chars += 1

        total_chars += len(hw.segmentation)

        if array_equal(np.array(seg_guess), np.array(hw.segmentation)):
            total_correct += 1

        total += 1

    for val in history:
        prediction, truth, latex = val
        print "*****************************************************"
        print "Prediction: {}".format(prediction)
        print "Truth: {}".format(truth)
        print "Latex: {}".format(latex)
        print "*****************************************************"

    print "OCR Count Min Distance: {}".format(ocr_count)
    return total_correct, total, total_correct_chars, total_chars, ocr_count


def segmentation_overlap_heuristic_with_ocr(recordings, use_pset4_cnn = True, use_pytorch_cnn = False):
    import cnn_from_pset4 as cnn
    import nn_with_inkml as pytorch_cnn
    import util
    import cPickle
    INTERSECTION_PROBABILITY_THRESHOLD = 0.9
    total_correct, total, total_correct_chars, total_chars = 0, 0, 0, 0
    history = []
    if use_pset4_cnn:
        # params, num_classes, y_map = cnn.main()
      #  y_map, num_classes = cnn.main()
      #  with open('cseg_y_map', 'w') as fp:
      #      cPickle.dump(y_map, fp)
      #  with open('cseg_num_classes', 'w') as fp:
      #      cPickle.dump(num_classes, fp)

        with open('cseg_y_map', 'rb') as fp:
            y_map = cPickle.load(fp)
        with open('cseg_num_classes', 'rb') as fp:
            num_classes = cPickle.load(fp)
        params = {}
        with open('regularized_W1', 'rb') as fp:
            params['W1'] = cPickle.load(fp)
        with open('regularized_b1', 'rb') as fp:
            params['b1'] = cPickle.load(fp)
        with open('regularized_W2', 'rb') as fp:
            params['W2'] = cPickle.load(fp)
        with open('regularized_b2', 'rb') as fp:
            params['b2'] = cPickle.load(fp)

        inv_map = {v: k for k, v in y_map.iteritems()}

    if use_pytorch_cnn:
        net, classes, y_map = pytorch_cnn.main()
        inv_map = {v: k for k, v in y_map.iteritems()}

    ocr_used = 0
    for so_far, hw in enumerate(recordings):
        strokes = hw.strokes
        pointlist = hw.get_pointlist()
        print "OVERLAP: File #{}".format(so_far)
        if hw == None:
            continue
        traces = literal_eval(hw.raw_data_json)
        # Get trace bounds
        trace_bounds = []
        for t in traces:
            max_x, min_x, max_y, min_y = float('-inf'), float('+inf'), float('-inf'), float('+inf')
            for d in t:
                y = d['y']
                x = d['x']
                if y > max_y:
                    max_y = y
                elif y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                elif x < min_x:
                    min_x = x
            trace_bounds.append((max_x, min_x, max_y, min_y))

        # Determine groupings:
        seg_guess = []
        current_group = [0]
        for i in range(1, len(trace_bounds)):

            for check_index in current_group:
                if i == check_index:
                    continue
              #  to_check = trace_bounds[check_index]
                # Check trace_bounds[i] for overlap with to_check
                distance = traceDistance(strokes[i], strokes[check_index])

               # print "min distance, ", distance
                if distance < 0.1:
                    current_group.append(i)
                    break
                elif use_pset4_cnn and abs(i - check_index) < 2:
                    group = [pointlist[i], pointlist[check_index]]
                    image_file = plot_traces(group, str(so_far) + '_' + str(i) + '_' + str(check_index))
                    array = np.asarray(Image.open(image_file).convert('L'))
                    data = [array.flatten()]
                    data = np.array(data, dtype=np.float128)
                    prediction = cnn.predict(data, params, num_classes)[0]
                    max_prediction = inv_map[np.argmax(prediction)]
                    top_predictions = np.argsort(prediction)[::-1][:1]
                    max_probability = prediction[top_predictions[0]]
                    summary = {}
                    if max_probability > INTERSECTION_PROBABILITY_THRESHOLD:
                        for symbol in util.commonly_missegmented_symbols:
                            if symbol not in y_map:
                                continue
                            id = y_map[symbol]
                            if id in top_predictions:
                                print "value of i: {}, to_check: {}, OCR for segmentation".format(i, check_index)
                                ocr_used += 1
                                current_group.append(i)
                                break

                elif use_pytorch_cnn and abs(i-check_index) < 2:
                    group = [pointlist[i], pointlist[check_index]]
                    image_file = plot_traces(group, str(so_far) + '_' + str(i) + '_' + str(check_index))
                    image = Image.open(image_file)
                    image_tensor = pytorch_cnn.transform(image)
                    net, classes = pytorch_cnn.main()
                    prediction = pytorch_cnn.test_single_example(image_tensor, net, classes)
                    top_five_predictions = np.argsort(prediction)[::-1][:1]
                    max_probability = 1 # top_predictions[0]
                    summary = {}
                    if max_probability > INTERSECTION_PROBABILITY_THRESHOLD:
                        for symbol in util.commonly_missegmented_symbols:
                            if symbol not in y_map:
                                continue
                            id = y_map[symbol]
                            if id in top_five_predictions:
                                print "value of i: {}, to_check: {}, OCR for segmentation".format(i, check_index)
                                ocr_used += 1
                                current_group.append(i)
                                break

                else:
                    seg_guess.append(current_group)
                    current_group = [i]
                    break
        if len(current_group) > 0:
            seg_guess.append(current_group)

        for entry in range(len(pointlist)):
            entry_found = False
            for group in seg_guess:
                if entry in group:
                    entry_found = True
                    break

            if not entry_found:
                seg_guess.append([entry])
    # print 'GUESS: ', seg_guess
    # print 'CORRECT: ', hw.segmentation

        history.append((sorted(seg_guess, key=itemgetter(0)), hw.segmentation))

        for g in seg_guess:
            if g in hw.segmentation: total_correct_chars += 1

        total_chars += len(hw.segmentation)

        if array_equal(np.array((sorted(seg_guess, key=itemgetter(0)))), np.array(hw.segmentation)):
            print "Equal"
            total_correct += 1

        total += 1



    for val in history:
        prediction, truth = val
        print "*****************************************************"
        print "Prediction: {}".format(prediction)
        print "Truth: {}".format(truth)
        print "*****************************************************"

    print "OCR used to segment {} times".format(ocr_used)
    print "Total correct: {}".format(total_correct)
    return total_correct, total, total_correct_chars, total_chars, ocr_used



def segmentation_overlap_heuristic_simple(recordings, use_pset4_cnn=True, use_pytorch_cnn=False):
    import cnn_from_pset4 as cnn
    import nn_with_inkml as pytorch_cnn
    import util
    import cPickle
    total_correct, total, total_correct_chars, total_chars = 0, 0, 0, 0
    history = []
    if use_pset4_cnn:
        # params, num_classes, y_map = cnn.main()
        #  y_map, num_classes = cnn.main()
        #  with open('cseg_y_map', 'w') as fp:
        #      cPickle.dump(y_map, fp)
        #  with open('cseg_num_classes', 'w') as fp:
        #      cPickle.dump(num_classes, fp)

        with open('cseg_y_map', 'rb') as fp:
            y_map = cPickle.load(fp)
        with open('cseg_num_classes', 'rb') as fp:
            num_classes = cPickle.load(fp)
        params = {}
        with open('regularized_W1', 'rb') as fp:
            params['W1'] = cPickle.load(fp)
        with open('regularized_b1', 'rb') as fp:
            params['b1'] = cPickle.load(fp)
        with open('regularized_W2', 'rb') as fp:
            params['W2'] = cPickle.load(fp)
        with open('regularized_b2', 'rb') as fp:
            params['b2'] = cPickle.load(fp)

        inv_map = {v: k for k, v in y_map.iteritems()}

    if use_pytorch_cnn:
        net, classes, y_map = pytorch_cnn.main()
        inv_map = {v: k for k, v in y_map.iteritems()}

    for so_far, hw in enumerate(recordings):
        strokes = hw.strokes
        pointlist = hw.get_pointlist()
        print "OVERLAP: File #{}".format(so_far)
        if hw == None:
            continue
        traces = literal_eval(hw.raw_data_json)
        # Get trace bounds
        trace_bounds = []
        for t in traces:
            max_x, min_x, max_y, min_y = float('-inf'), float('+inf'), float('-inf'), float('+inf')
            for d in t:
                y = d['y']
                x = d['x']
                if y > max_y:
                    max_y = y
                elif y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                elif x < min_x:
                    min_x = x
            trace_bounds.append((max_x, min_x, max_y, min_y))

        # Determine groupings:
        seg_guess = []
        current_group = [0]
        for i in range(1, len(trace_bounds)):

            for check_index in current_group:
                if i == check_index:
                    continue
                to_check = trace_bounds[check_index]
                # Check trace_bounds[i] for overlap with to_check
                if trace_bounds[i][1] < to_check[0]:
                    current_group.append(i)
                    break
                else:
                    seg_guess.append(current_group)
                    current_group = [i]
                    break
        if len(current_group) > 0:
            seg_guess.append(current_group)
            # print 'GUESS: ', seg_guess
            # print 'CORRECT: ', hw.segmentation

        for entry in range(len(pointlist)):
            entry_found = False
            for group in seg_guess:
                if entry in group:
                    entry_found = True
                    break

            if not entry_found:
                seg_guess.append([entry])

        history.append((sorted(seg_guess, key=itemgetter(0)), hw.segmentation))
        for g in seg_guess:
            if g in hw.segmentation: total_correct_chars += 1

        total_chars += len(hw.segmentation)

        if array_equal(np.array(seg_guess), np.array(hw.segmentation)):
            total_correct += 1

        total += 1



    for val in history:
        prediction, truth = val
        print "*****************************************************"
        print "Prediction: {}".format(prediction)
        print "Truth: {}".format(truth)
        print "*****************************************************"

    return total_correct, total, total_correct_chars, total_chars, 0


def segmentation_baseline_with_ocr(recordings, use_pset4_cnn=True, use_pytorch_cnn=False):
    import cnn_from_pset4 as cnn
    import nn_with_inkml as pytorch_cnn
    import util
    import cPickle
    total_correct, total, total_correct_chars, total_chars = 0, 0, 0, 0
    history = []
    INTERSECTION_PROBABILITY_THRESHOLD = 0.8
    if use_pset4_cnn:
        # params, num_classes, y_map = cnn.main()
        #  y_map, num_classes = cnn.main()
        #  with open('cseg_y_map', 'w') as fp:
        #      cPickle.dump(y_map, fp)
        #  with open('cseg_num_classes', 'w') as fp:
        #      cPickle.dump(num_classes, fp)

        with open('cseg_y_map', 'rb') as fp:
            y_map = cPickle.load(fp)
        with open('cseg_num_classes', 'rb') as fp:
            num_classes = cPickle.load(fp)
        params = {}
        with open('regularized_W1', 'rb') as fp:
            params['W1'] = cPickle.load(fp)
        with open('regularized_b1', 'rb') as fp:
            params['b1'] = cPickle.load(fp)
        with open('regularized_W2', 'rb') as fp:
            params['W2'] = cPickle.load(fp)
        with open('regularized_b2', 'rb') as fp:
            params['b2'] = cPickle.load(fp)

        inv_map = {v: k for k, v in y_map.iteritems()}

    ocr_used = 0
    for so_far, hw in enumerate(recordings):
        strokes = hw.strokes
        pointlist = hw.get_pointlist()
        print "OVERLAP: File #{}".format(so_far)
        if hw == None:
            continue
        traces = literal_eval(hw.raw_data_json)
        # Get trace bounds
        trace_bounds = []
        for t in traces:
            max_x, min_x, max_y, min_y = float('-inf'), float('+inf'), float('-inf'), float('+inf')
            for d in t:
                y = d['y']
                x = d['x']
                if y > max_y:
                    max_y = y
                elif y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                elif x < min_x:
                    min_x = x
            trace_bounds.append((max_x, min_x, max_y, min_y))

        # Determine groupings:
        seg_guess = []
        current_group = [0]
        for i in range(1, len(trace_bounds)):
            for check_index in current_group:
                if i == check_index:
                    continue
                if use_pset4_cnn and abs(i - check_index) < 2:
                    group = [pointlist[i], pointlist[check_index]]
                    image_file = plot_traces(group, str(so_far) + '_' + str(i) + '_' + str(check_index))
                    array = np.asarray(Image.open(image_file).convert('L'))
                    data = [array.flatten()]
                    data = np.array(data, dtype=np.float128)
                    prediction = cnn.predict(data, params, num_classes)[0]
                    top_predictions = np.argsort(prediction)[::-1][:1]
                    max_probability = prediction[top_predictions[0]]
                    summary = {}
                    if max_probability > INTERSECTION_PROBABILITY_THRESHOLD:
                        print "value of i: {}, to_check: {}, OCR for segmentation".format(i, check_index)
                        ocr_used += 1
                        current_group.append(i)
                        break
                            # print "************MAX PREDICTION: {} **** INTERSECTION PROBABILITIES: {}".format(max_prediction, summary)
                elif use_pytorch_cnn and abs(i-check_index) < 2:
                    group = [pointlist[i], pointlist[check_index]]
                    image_file = plot_traces(group, str(so_far) + '_' + str(i) + '_' + str(check_index))
                    image = Image.open(image_file)
                    image_tensor = pytorch_cnn.transform(image)
                    net, classes = pytorch_cnn.main()
                    prediction = pytorch_cnn.test_single_example(image_tensor, net, classes)
                    top_five_predictions = np.argsort(prediction)[::-1][:1]
                    max_probability = top_predictions[0]
                    summary = {}
                    if max_probability > INTERSECTION_PROBABILITY_THRESHOLD:
                        for symbol in util.commonly_missegmented_symbols:
                            if symbol not in y_map:
                                continue
                            id = y_map[symbol]
                            if id in top_five_predictions:
                                print "value of i: {}, to_check: {}, OCR for segmentation".format(i, check_index)
                                ocr_used += 1
                                current_group.append(i)
                                break
                else:
                    seg_guess.append(current_group)
                    current_group = [i]
                    break

        if len(current_group) > 0:
            seg_guess.append(current_group)
            # print 'GUESS: ', seg_guess
            # print 'CORRECT: ', hw.segmentation

        for entry in range(len(pointlist)):
            entry_found = False
            for group in seg_guess:
                if entry in group:
                    entry_found = True
                    break

            if not entry_found:
                seg_guess.append([entry])

        history.append((sorted(seg_guess, key=itemgetter(0)), hw.segmentation))
        for g in seg_guess:
            if g in hw.segmentation: total_correct_chars += 1

        total_chars += len(hw.segmentation)

        if array_equal(np.array(sorted(seg_guess, key=itemgetter(0))), np.array(hw.segmentation)):
            total_correct += 1

        total += 1

    for val in history:
        prediction, truth = val
        print "*****************************************************"
        print "Prediction: {}".format(prediction)
        print "Truth: {}".format(truth)
        print "*****************************************************"

    return total_correct, total, total_correct_chars, total_chars, ocr_used

def segmentation_baseline(recordings, use_pset4_cnn=True, use_pytorch_cnn=False):
    total_correct, total, total_correct_chars, total_chars = 0, 0, 0, 0
    for i, hw in enumerate(recordings):
        print  "BASELINE: #{}".format(i)
        if hw == None:
            continue
        traces = literal_eval(hw.raw_data_json)
        seg_guess = []
        for index, t in enumerate(traces):
            seg_guess.append([index])

        for g in seg_guess:
            if g in hw.segmentation: total_correct_chars += 1

        total_chars += len(hw.segmentation)

        if array_equal(np.array(seg_guess), np.array(hw.segmentation)):
            total_correct += 1

        total += 1

    return total_correct, total, total_correct_chars, total_chars


from util import read_files_segmentation
from util import svm_linear_train

def trainSVM(path, train_dir):
    TRAINING_X, TRAINING_Y = read_files_segmentation(path, path + train_dir)
    clf = svm_linear_train(TRAINING_X, TRAINING_Y)
    y_map = {}
    return clf, y_map

from inkml import read_equations

def main():
    print "HELLO"
    path = '/Users/norahborus/Documents/DATA/COMBINED/'
    path = '/Users/norahborus/Documents/DATA/training_data/'
    train_preprocessed = 'TRAIN_preprocessed'
    test_preprocessed = 'TEST_preprocessed'
   # training_folders = [path, "TRAIN/", "TEST/"]
    folders = [path, "CROHME_training_2011/", "TrainINKML_2013/"]
   # clf, y_map = trainSVM(path, train_preprocessed)
    print "BEFORE RECORDINGS"
    recordings = read_equations(folders, 1,2)
    print "Recordings done"

    total_correct_1, total_1, total_correct_chars_1, total_chars_1 = segmentation_baseline(
        recordings)

    print "*****************************************************************"
    print 'BASELINE FIRST ACCURACY --> ', 100. * total_correct_1 / total_1
    print 'BASELINE: SECOND ACCURACY --> ', 100. * total_correct_chars_1 / total_chars_1
    print "*****************************************************************"

    total_correct_2, total_2, total_correct_chars_2, total_chars_2, ocr_count = segmentation_baseline_with_ocr(recordings)
    print "*****************************************************************"
    print 'BASELINE WITH OCR: FIRST ACCURACY --> ', 100. * total_correct_2 / total_2
    print 'BASELINE WITH OCR: SECOND ACCURACY --> ', 100. * total_correct_chars_2 / total_chars_2
    print '# TIMES OCR USED IN SEGMENTATION   --->', ocr_count
    print "*****************************************************************"

    print "*****************************************************************"
    print "*****************************************************************"
    print "SUMMARY:"
    print "NUM FILES: {}".format(len(recordings))
    print 'BASELINE FIRST ACCURACY --> ', 100. * total_correct_1 / total_1
    print 'BASELINE: SECOND ACCURACY --> ', 100. * total_correct_chars_1 / total_chars_1

    print 'BASELINE WITH OCR:: FIRST ACCURACY --> ', 100. * total_correct_2 / total_2
    print 'BASELINE WITH OCR:: SECOND ACCURACY --> ', 100. * total_correct_chars_2 / total_chars_2
    print '# TIMES OCR USED IN SEGMENTATION   --->', ocr_count

    print "*****************************************************************"
    print "*****************************************************************"





'''

    total_correct_3, total_3, total_correct_chars_3, total_chars_3 = segmentation_overlap_heuristic_simple(recordings)
    print "*****************************************************************"
    print "*****************************************************************"
    print 'OVERLAP HEURISTIC WITHOUT OCR: FIRST ACCURACY --> ', 100. * total_correct_3 / total_3
    print 'OVERLAP HEURISTIC WITHOUT OCR: SECOND ACCURACY --> ', 100. * total_correct_chars_3 / total_chars_3
    print "*****************************************************************"


    total_correct_2, total_2, total_correct_chars_2, total_chars_2, ocr_count = segmentation_svm_min_distance(
        recordings)

    print "*****************************************************************"
    print 'MIN DISTANCE WITH OCR: FIRST ACCURACY --> ', 100. * total_correct_2 / total_2
    print 'MIN DISTANCE WITH OCR: SECOND ACCURACY --> ', 100. * total_correct_chars_2 / total_chars_2
    print '# TIMES OCR USED IN SEGMENTATION   --->', ocr_count
    print "*****************************************************************"


    print "*****************************************************************"
    print "*****************************************************************"
    print "SUMMARY:"
    print 'BASELINE FIRST ACCURACY --> ', 100. * total_correct_1 / total_1
    print 'BASELINE: SECOND ACCURACY --> ', 100. * total_correct_chars_1 / total_chars_1

    print 'MIN DISTANCE WITH OCR: FIRST ACCURACY --> ', 100. * total_correct_2 / total_2
    print 'MIN DISTANCE WITH OCR: SECOND ACCURACY --> ', 100. * total_correct_chars_2 / total_chars_2


    print 'OVERLAP HEURISTIC WITHOUT OCR: FIRST ACCURACY --> ', 100. * total_correct_3 / total_3
    print 'OVERLAP HEURISTIC WITHOUT OCR: SECOND ACCURACY --> ', 100. * total_correct_chars_3 / total_chars_3

'''

if __name__ == '__main__':
    print "CSEG"
    main()
