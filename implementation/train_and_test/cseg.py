from __future__ import division
import sys, os, random
from inkml import read
from ast import literal_eval
import numpy as np
from numpy import array_equal

from collections import defaultdict
from operator import itemgetter
import itertools
from sklearn import datasets, svm, metrics
import cPickle

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
def min_distance_segment(so_far, hw, nn_params=None, pytorch_params=None, use_nn =False, use_cnn= False, ):
    import nn as nn
    import cnn as cnn
    min_global = 5
    strokes = hw.strokes
    pointlist = hw.get_pointlist()
   # print len(pointlist)
    neighbors = defaultdict(list)
    ocr_used = 0
    INTERSECTION_PROBABILITY_THRESHOLD = 0.7

    if use_nn:
        y_map = nn_params['y_map']
        num_classes = nn_params['num_classes']
        inv_map = nn_params['inv_map']
        params = nn_params['params']

    if use_cnn:
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
            elif use_nn and abs(i - index) < 2:
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
            elif use_cnn:
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



def segmentation_svm_min_distance(recordings, use_nn = False, use_cnn = True):
    import nn as nn
    import cnn as cnn
    import util

    if use_nn:
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

    if use_cnn:

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


def segmentation_overlap_heuristic_with_ocr(recordings, use_nn = True, use_cnn = False):
    import nn as nn
    import cnn as cnn
    import util
    import cPickle
    INTERSECTION_PROBABILITY_THRESHOLD = 0.9
    total_correct, total, total_correct_chars, total_chars = 0, 0, 0, 0
    history = []
    if use_nn:
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

    if use_cnn:
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
                elif use_nn and abs(i - check_index) < 2:
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

                elif use_cnn and abs(i-check_index) < 2:
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



def segmentation_overlap_heuristic_simple(recordings, use_nn=True, use_cnn=False):
    import nn as nn
    import cnn as cnn
    import util
    import cPickle
    total_correct, total, total_correct_chars, total_chars = 0, 0, 0, 0
    history = []
    if use_nn:
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

    if use_cnn:
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


def segmentation_baseline_with_ocr(recordings, use_nn=True, use_cnn=False):
    import nn as nn
    import cnn as cnn
    import util
    import cPickle
    total_correct, total, total_correct_chars, total_chars = 0, 0, 0, 0
    history = []
    INTERSECTION_PROBABILITY_THRESHOLD = 0.8
    if use_nn:
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
                if use_nn and abs(i - check_index) < 2:
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
                elif use_cnn and abs(i-check_index) < 2:
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


def segmentation_baseline(recordings, use_nn=True, use_cnn=False):
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

# # For each expression, generates all possible assignments (with size up to 4).
# # For each assignment, tries to classify and chooses the assignment with the
# # highest score.
# def segmentation_ocr_dp(recordings, use_pset4_cnn=True, use_pytorch_cnn=True):
#     import cnn_from_pset4 as cnn
#     import nn_with_grayscale as pytorch_cnn
#     import util
#     import cPickle
#     from collections import defaultdict
#     import copy

#     use_class_nn = True
##    Only test one example
#     recordings = recordings[:1]
#     total_correct, total, total_correct_chars, total_chars = 0, 0, 0, 0
#     total_correct_short, total_correct_long = 0, 0
#     history = []
#     max_assignment_len_with_scale, max_assignment_len = 4, 4
#     train_from_scrach = False
#     if train_from_scrach:
#         net, classes, y_map = pytorch_cnn.main()
#         with open('cnn_params.pkl', 'w') as f:
#             cPickle.dump([net, classes, y_map], f)
#     else:
#         with open('cnn_params.pkl') as f:
#             net, classes, y_map = cPickle.load(f)
#     for so_far, hw in enumerate(recordings):
#         max_assignment_len_with_scale = max_assignment_len
#         complete_assignments_list, complete_assignments_scores = [], []
#         groups_scores_map = defaultdict(float)
#         strokes = hw.strokes # Unnecessary?
#         pointlist = hw.get_pointlist()
#         print "DP: File #{}".format(so_far)
#         if hw == None: continue
#         traces = literal_eval(hw.raw_data_json)
#         if len(traces) > 25:
#             max_assignment_len_with_scale = min(max_assignment_len_with_scale, 1)
#         elif len(traces) > 20:
#             max_assignment_len_with_scale = min(max_assignment_len_with_scale, 2)
#         traces_numbers = range(0, len(traces))
#         if len(traces) < 1: continue

#         # Recursive function that generates all possible ordered complete assignments
#         # with max group len up to max_assignment_len.
#         def generate_complete_assignments(start_index, curr_assignment):
#             if len(curr_assignment[-1]) > max_assignment_len_with_scale: return # Base case: last group in assignment too long - invalid
#             if start_index >= len(traces): # Complete assignment - add to complete assignments
#                 complete_assignments_list.append(curr_assignment)
#                 return
#             last_assignment_append = (copy.deepcopy(curr_assignment)) # Adds trace to the last group
#             last_assignment_append[-1].append(traces_numbers[start_index])
#             generate_complete_assignments(start_index + 1, last_assignment_append)
#             assignment_new_group = (copy.deepcopy(curr_assignment))
#             assignment_new_group.append([traces_numbers[start_index]]) # Creates a new group and add trace to it
#             generate_complete_assignments(start_index + 1, assignment_new_group)

#         # Based on a list of assignments, assigns a classification score for each assignment
#         def get_assignments_scores():
#             for complete_assignment_ind, complete_assignment in enumerate(complete_assignments_list):
#                 assignment_total_score = 0
#                 for group_ind, group in enumerate(complete_assignment):
#                     group_score = get_group_score(group, traces, complete_assignment_ind, group_ind)
#                     assignment_total_score += group_score
#                     print "group: ", group, "score: ", group_score

#                 # print "score: ", assignment_total_score / len(complete_assignment)
#                 complete_assignments_scores.append(assignment_total_score / len(complete_assignment))
#                 print "assignment score: ", (assignment_total_score / len(complete_assignment))
#                 print "Done with: equation", so_far, "assignment " , complete_assignment_ind, " out of ", len(complete_assignments_list)

#         # Calculates the classification score for a group of traces that is believed to be one char.
#         def get_group_score(group, traces, assignment_index, group_ind):
#             if tuple(group) in groups_scores_map:
#                 return groups_scores_map[tuple(group)]
#             group_traces = [traces[i] for i in group]
#             image_file = plot_traces(group_traces, str(so_far) + '_' +str(assignment_index) + '_' + str(group_ind) + '_' + "group")
#             if use_class_nn == False:
#                 image = Image.open(image_file).convert('L')
#                 image_tensor = pytorch_cnn.transform(image)
#                 prediction = pytorch_cnn.test_single_example(image_tensor, net, classes)[0]
#                 group_score = prediction[-1]
#                 groups_scores_map[tuple(group)] = group_score
#             else:
#                 with open('cseg_y_map', 'rb') as fp:
#                     y_map = cPickle.load(fp)
#                 with open('cseg_num_classes', 'rb') as fp:
#                     num_classes = cPickle.load(fp)
#                 with open('params.pkl', 'rb') as fp:
#                     params = cPickle.load(fp)

#                 array = np.asarray(Image.open(image_file).convert('L'))
#                 data = [array.flatten()]
#                 data = np.array(data, dtype=np.float128)
#                 prediction = cnn.predict(data, params, num_classes)[0]
#                 print prediction
#                 return prediction[y_map['.NG']]
#                 # top_predictions = np.argsort(prediction)[::-1][:1]
#                 # max_probability = prediction[top_predictions[0]]
#             # print prediction
#             # max_score = max(prediction)
#             # groups_scores_map[tuple(group)] = max_score
#             # return max_score

            
#             return group_score

#         generate_complete_assignments(1, [[traces_numbers[0]]])
#         get_assignments_scores()
#         # print "assignment scores", complete_assignments_scores
#         best_assignment_ind = complete_assignments_scores.index(max(complete_assignments_scores))
#         best_assignment = complete_assignments_list[best_assignment_ind]
        
#         #Calculate accuracy for single chars and whole equation
#         sorted_true_segmentation = [sorted(elem) for elem in hw.segmentation]
#         for group in best_assignment:
#             if sorted(group) in sorted_true_segmentation:
#                 total_correct_chars += 1
#                 if len(group) == 1:
#                     total_correct_short += 1
#                 else:
#                     total_correct_long += 1
        
#         total_chars += len(hw.segmentation)
#         if array_equal(np.array(best_assignment), np.array(hw.segmentation)):
#             total_correct += 1

#         total += 1
#         print "done with expression: ", so_far
#         print "chosen grouping: ", best_assignment, "true groupings: ", np.array(hw.segmentation)
#     print "total correct long: ", total_correct_long, "total correct short: ", total_correct_short
#     return total_correct, total, total_correct_chars, total_chars

import util
import cPickle
from collections import defaultdict
import copy
from operator import itemgetter

def get_group_score(group, assignment_index, so_far, group_ind, groups_scores_map, traces):
    use_class_nn = True
    if groups_scores_map and tuple(group) in groups_scores_map:
        return groups_scores_map[tuple(group)]
    print "group: ", group
    group_traces = [traces[i] for i in group]
    image_file = plot_traces(group_traces, str(so_far) + '_' +str(assignment_index) + '_' + str(group_ind) + '_' + "group")
    if use_class_nn == False:
        image = Image.open(image_file).convert('L')
        image_tensor = pytorch_cnn.transform(image)
        prediction = pytorch_cnn.test_single_example(image_tensor, net, classes)[0]
        group_score = prediction[-1]
    else:
        with open('cseg_y_map', 'rb') as fp:
            y_map = cPickle.load(fp)
        with open('cseg_num_classes', 'rb') as fp:
            num_classes = cPickle.load(fp)
        with open('params.pkl', 'rb') as fp:
            params = cPickle.load(fp)

        array = np.asarray(Image.open(image_file).convert('L'))
        data = [array.flatten()]
        data = np.array(data, dtype=np.float128)
        prediction = cnn.predict(data, params, num_classes)[0]
        group_score = np.log(prediction[y_map['.NG']])
    groups_scores_map[tuple(group)] = group_score
    print "group score: ", group_score
    return group_score

# Recursive function that generates all possible ordered complete assignments
# with max group len up to max_assignment_len.
def extend_possible_assignments(partial_assignments, complete_assignments_list, complete_assignments_scores, groups_scores_map, traces, max_assignment_len, so_far):
    new_partial_assignments = []
    for partial_assignment_ind, partial_assignment_list in enumerate(partial_assignments):
        score_so_far, counter, partial_assignment = partial_assignment_list[0], partial_assignment_list[1], partial_assignment_list[2]
        if len(partial_assignment) <= 0:
            index_to_assign = 0
        else: 
            # next index to assign is the last index in the last group + 1
            index_to_assign = partial_assignment[-1][-1] + 1
        
        # This assignment is complete - add it to complete assignments and we're done processing
        if index_to_assign >= len(traces):
            complete_assignments_list.append(partial_assignment)
            complete_assignments_scores.append(score_so_far / counter)
            print "complete_assignments_list: ", complete_assignments_list
            print "complete_assignments_scores: ", complete_assignments_scores
        else:
            for group_len in range(max_assignment_len):
                if index_to_assign + group_len + 1 > len(traces): continue
                next_group = [index_to_assign + curr_elem for curr_elem in range(group_len + 1)]
                if partial_assignment:
                    new_partial_assignment = copy.deepcopy(partial_assignment)
                    new_partial_assignment.append(next_group)
                else:
                    new_partial_assignment = [next_group]
                next_group_score = get_group_score(next_group, partial_assignment_ind, so_far, group_len, groups_scores_map, traces)
                new_partial_assignments.append([score_so_far + next_group_score, counter + 1, new_partial_assignment, (score_so_far + next_group_score) / (counter + 1)])
    
    # Update the list of partial assignments with the extended assignments
    return new_partial_assignments

# Keep only k total assignments
def prune(k, complete_assignments_list, partial_assignments):
    num_to_keep = k - len(complete_assignments_list)
    if num_to_keep < 1: return
    partial_assignments = sorted(partial_assignments, key=itemgetter(3))
    return partial_assignments[-num_to_keep:]


# For each expression, generates all possible assignments (with size up to 4).
# For each assignment, tries to classify and chooses the assignment with the
# highest score.
def segmentation_ocr_beam_search(recordings, use_pset4_cnn=True, use_pytorch_cnn=True):

    recordings = recordings[:5]
    total_correct, total, total_correct_chars, total_chars = 0, 0, 0, 0
    total_correct_short, total_correct_long = 0, 0
    max_assignment_len, k = 4, 3
    groups_scores_map = defaultdict(float)
    
    # Load the NN parameters (train or load pretrained)
    train_from_scrach = False
    if train_from_scrach:
        net, classes, y_map = pytorch_cnn.main()
        with open('cnn_params.pkl', 'w') as f:
            cPickle.dump([net, classes, y_map], f)
    else:
        with open('cnn_params.pkl') as f:
            net, classes, y_map = cPickle.load(f)

    # For each expression - find segmentation
    for so_far, hw in enumerate(recordings):
        print "DP beam searc: File #{}".format(so_far)
        # partial_assignments is a list where each sublist includes sum_score, group_counter, list of groups (assignment), avg_score
        partial_assignments, complete_assignments_list, complete_assignments_scores = [[0.0, 0, [], 0.0]], [], []
        # Store scores of groups that have already been calculated
        groups_scores_map = defaultdict(float)
        if hw == None: continue
        # List of all traces (strokes) for the expression
        traces = literal_eval(hw.raw_data_json)
        traces_numbers = range(0, len(traces))
        if len(traces) < 1: continue
        print "traces: ",len(traces)
        # Once we have all complete assignments - pick the best one
        while len(complete_assignments_list) < k and partial_assignments:
            partial_assignments = extend_possible_assignments(partial_assignments, complete_assignments_list, complete_assignments_scores, groups_scores_map, traces, max_assignment_len, so_far)
            partial_assignments = prune(k, complete_assignments_list, partial_assignments)

        best_assignment_ind = complete_assignments_scores.index(max(complete_assignments_scores))
        best_assignment = complete_assignments_list[best_assignment_ind]
        
        #Calculate accuracy for single chars and whole equation
        sorted_true_segmentation = [sorted(elem) for elem in hw.segmentation]
        for group in best_assignment:
            if sorted(group) in sorted_true_segmentation:
                total_correct_chars += 1
                if len(group) == 1:
                    total_correct_short += 1
                else:
                    total_correct_long += 1
        
        total_chars += len(hw.segmentation)
        if array_equal(np.array(best_assignment), np.array(hw.segmentation)):
            total_correct += 1

        total += 1
        print "done with expression: ", so_far
        print "chosen grouping: ", best_assignment, "true groupings: ", np.array(hw.segmentation)
    print "total correct long: ", total_correct_long, "total correct short: ", total_correct_short
    return total_correct, total, total_correct_chars, total_chars

def generate_random_GC_examples(recordings):
    from sets import Set
    import random

    for so_far, hw in enumerate(recordings[-70:-50]):
        sorted_true_segmentation = [sorted(elem) for elem in hw.segmentation]
        used_groups = Set([])
        group_size = 1
        print "DP generates images: File #{}".format(so_far)
        if hw == None: continue
        traces = literal_eval(hw.raw_data_json)
        trace_numbers = range(0, len(traces))
        # Generate single garbage characters
        for j in range(min(len(traces), 10)):
            rand_group_ind = random.sample(trace_numbers, 1)
            print "rand_group_ind: ", rand_group_ind
            if tuple(rand_group_ind) in used_groups or rand_group_ind in sorted_true_segmentation: continue
            traces_group = [traces[i] for i in rand_group_ind]
            image_file = plot_traces(traces_group, 'garbage_class_' +  str(so_far) + '_' + str(j) + '_size1' + '.inkml.GC')
            used_groups.add(tuple(rand_group_ind))

        # Generate pairs garbage class
        for j in range(min(len(traces), 10)):
            rand_group_ind = random.sample(trace_numbers, 1)
            if rand_group_ind[0] == len(traces) - 1: continue
            rand_group_ind = [rand_group_ind[0], rand_group_ind[0] + 1]
            if tuple(rand_group_ind) in used_groups or rand_group_ind in sorted_true_segmentation: continue
            traces_group = [traces[i] for i in rand_group_ind]
            image_file = plot_traces(traces_group, 'garbage_class_' +  str(so_far) + '_' + str(j) + '_size2' + '.inkml.GC')
            used_groups.add(tuple(rand_group_ind))

        for j in range(min(len(traces), 10)):
            # 3 elements garbage
            rand_group_ind = random.sample(trace_numbers, 1)
            if rand_group_ind[0] >= len(traces) - 2: continue
            rand_group_ind = [rand_group_ind[0], rand_group_ind[0] + 1, rand_group_ind[0] + 2]
            if tuple(rand_group_ind) in used_groups or rand_group_ind in sorted_true_segmentation: continue
            traces_group = [traces[i] for i in rand_group_ind]
            image_file = plot_traces(traces_group, 'garbage_class_' +  str(so_far) + '_' + str(j) + '_size3' + '.inkml.GC')
            used_groups.add(tuple(rand_group_ind))

        for j in range(min(len(traces), 10)):
            # 4 elements garbage
            rand_group_ind = random.sample(trace_numbers, 1)
            if rand_group_ind[0] >= len(traces) - 3: continue
            rand_group_ind = [rand_group_ind[0], rand_group_ind[0] + 1, rand_group_ind[0] + 2, rand_group_ind[0] + 3]
            if tuple(rand_group_ind) in used_groups or rand_group_ind in sorted_true_segmentation: continue
            traces_group = [traces[i] for i in rand_group_ind]
            image_file = plot_traces(traces_group, 'garbage_class_' +  str(so_far) + '_' + str(j) + '_size4' + '.inkml.GC')
            used_groups.add(tuple(rand_group_ind))

from util import read_files_segmentation
from util import svm_linear_train

def trainSVM(path, train_dir):
    TRAINING_X, TRAINING_Y = read_files_segmentation(path, path + train_dir)
    clf = svm_linear_train(TRAINING_X, TRAINING_Y)
    y_map = {}
    return clf, y_map

from inkml import read_equations

def main():
    read_recordings_from_scrach = False
    print "HELLO"
    # path = '/Users/amit/project/DATA/COMBINED/'
    path = '/Users/amit/project/DATA/training_data/'
    train_preprocessed = 'TRAIN_preprocessed'
    test_preprocessed = 'TEST_preprocessed'
   # training_folders = [path, "TRAIN/", "TEST/"]
    folders = [path, "CROHME_training_2011/", "TrainINKML_2013/"]
   # clf, y_map = trainSVM(path, train_preprocessed)
    print "BEFORE RECORDINGS"
    if read_recordings_from_scrach:
        recordings = read_equations(folders, 1,2)
        with open('recordings.pkl', 'w') as f:
            cPickle.dump(recordings, f)
    else:
        with open('recordings.pkl') as f:
            recordings = cPickle.load(f)
    print "Recordings done"

    # generate_random_GC_examples(recordings)    
    total_correct_dp, total_dp, total_correct_chars_dp, total_chars_dp = segmentation_ocr_beam_search(recordings)
    print "*****************************************************************"
    print 'DP ocr: FIRST ACCURACY --> ', 100. * total_correct_dp / total_dp
    print 'DP ocr: SECOND ACCURACY --> ', 100. * total_correct_chars_dp / total_chars_dp
    print "*****************************************************************"
    # total_correct_dp, total_dp, total_correct_chars_dp, total_chars_dp = segmentation_ocr_dp(recordings)
    # print "*****************************************************************"
    # print 'DP ocr: FIRST ACCURACY --> ', 100. * total_correct_dp / total_dp
    # print 'DP ocr: SECOND ACCURACY --> ', 100. * total_correct_chars_dp / total_chars_dp
    # print "*****************************************************************"

    # print "*****************************************************************"
    # print "*****************************************************************"
    # print "SUMMARY:"
    # print "NUM FILES: {}".format(len(recordings))
    # print 'BASELINE FIRST ACCURACY --> ', 100. * total_correct_1 / total_1
    # print 'BASELINE: SECOND ACCURACY --> ', 100. * total_correct_chars_1 / total_chars_1



    # total_correct_1, total_1, total_correct_chars_1, total_chars_1 = segmentation_baseline(
    #     recordings)

    # print "*****************************************************************"
    # print 'BASELINE FIRST ACCURACY --> ', 100. * total_correct_1 / total_1
    # print 'BASELINE: SECOND ACCURACY --> ', 100. * total_correct_chars_1 / total_chars_1
    # print "*****************************************************************"

    # total_correct_2, total_2, total_correct_chars_2, total_chars_2, ocr_count = segmentation_baseline_with_ocr(recordings)
    # print "*****************************************************************"
    # print 'BASELINE WITH OCR: FIRST ACCURACY --> ', 100. * total_correct_2 / total_2
    # print 'BASELINE WITH OCR: SECOND ACCURACY --> ', 100. * total_correct_chars_2 / total_chars_2
    # print '# TIMES OCR USED IN SEGMENTATION   --->', ocr_count
    # print "*****************************************************************"

    # print 'DP ocr: FIRST ACCURACY --> ', 100. * total_correct_dp / total_dp
    # print 'DP ocr: SECOND ACCURACY --> ', 100. * total_correct_chars_dp / total_chars_dp
    # print 'BASELINE WITH OCR:: FIRST ACCURACY --> ', 100. * total_correct_2 / total_2
    # print 'BASELINE WITH OCR:: SECOND ACCURACY --> ', 100. * total_correct_chars_2 / total_chars_2
    # print '# TIMES OCR USED IN SEGMENTATION   --->', ocr_count

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
