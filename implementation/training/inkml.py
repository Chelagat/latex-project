#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import svm
import json
import signal
import sys
import logging
import numpy as np
import matplotlib
from sys import argv
matplotlib.use("Agg")
import matplotlib.pyplot as pl
import pickle
from collections import defaultdict

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

import random
from xml.dom.minidom import parseString

# hwrt modules
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import handwritten_data


# from __init__ import formula_to_dbid


def beautify_xml(path):
    """
    Beautify / pretty print XML in `path`.

    Parameters
    ----------
    path : str

    Returns
    -------
    str
    """
    with open(path) as f:
        content = f.read()

    pretty_print = lambda data: '\n'.join([line for line in
                                           parseString(data)
                                          .toprettyxml(indent=' ' * 2)
                                          .split('\n')
                                           if line.strip()])
    return pretty_print(content)


def normalize_symbol_name(symbol_name):
    """
    Change symbol names to a version which is known by write-math.com

    Parameters
    ----------
    symbol_name : str

    Returns
    -------
    str
    """
    if symbol_name == '\\frac':
        return '\\frac{}{}'
    elif symbol_name == '\\sqrt':
        return '\\sqrt{}'
    elif symbol_name in ['&lt;', '\lt']:
        return '<'
    elif symbol_name in ['&gt;', '\gt']:
        return '>'
    elif symbol_name == '{':
        return '\{'
    elif symbol_name == '}':
        return '\}'
    return symbol_name


def read(folder, filepath, short_filename, directory):
    print filepath
    if filepath[-2:] == 'lg':
        return None
    import xml.etree.ElementTree
    try:
        root = xml.etree.ElementTree.parse(filepath).getroot()
    except:
        "error"
        return None
    # Get the raw data
    recording = []
    strokes = sorted(root.findall('{http://www.w3.org/2003/InkML}trace'),
                     key=lambda child: int(child.attrib['id']))
    time = 0
    for stroke in strokes:
        stroke = stroke.text.strip().split(',')
        stroke = [point.strip().split(' ') for point in stroke]
        if len(stroke[0]) == 3:
            stroke = [{'x': float(x), 'y': float(y), 'time': float(t)}
                      for x, y, t in stroke]
        else:
            stroke = [{'x': float(x), 'y': float(y)} for x, y in stroke]
            new_stroke = []
            for p in stroke:
                new_stroke.append({'x': p['x'], 'y': p['y'], 'time': time})
                time += 20
            stroke = new_stroke
            time += 200
        recording.append(stroke)

    # Get LaTeX
    formula_in_latex = None
    annotations = root.findall('{http://www.w3.org/2003/InkML}annotation')
    for annotation in annotations:
        if annotation.attrib['type'] == 'truth':
            formula_in_latex = annotation.text
    hw = handwritten_data.HandwrittenData(json.dumps(recording), formula_in_latex=formula_in_latex,
                                          filename=short_filename, filepath=folder[0] + directory,
                                          raw_data_id=directory + short_filename)
    for annotation in annotations:
        if annotation.attrib['type'] == 'writer':
            hw.writer = annotation.text
        elif annotation.attrib['type'] == 'category':
            hw.category = annotation.text
        elif annotation.attrib['type'] == 'expression':
            hw.expression = annotation.text

    # Get segmentation
    segmentation = []
    trace_groups = root.findall('{http://www.w3.org/2003/InkML}traceGroup')
    if len(trace_groups) != 1:
        raise Exception('Malformed InkML',
                        ('Exactly 1 top level traceGroup expected, found %i. '
                         '(%s) - probably no ground truth?') %
                        (len(trace_groups), filepath))
    trace_group = trace_groups[0]
    symbol_stream = []  # has to be consistent with segmentation
    baseline_parsed = {}
    for tg in trace_group.findall('{http://www.w3.org/2003/InkML}traceGroup'):
        annotations = tg.findall('{http://www.w3.org/2003/InkML}annotation')
        # anno_xml = tg.findall('{http://www.w3.org/2003/InkML}annotationXML')
        if len(annotations) != 1:
            raise ValueError("%i annotations found for '%s'." %
                             (len(annotations), filepath))

        value = annotations[0].text
        trace_views = tg.findall('{http://www.w3.org/2003/InkML}traceView')
        symbol = []
        trace_ids = []
        for traceView in trace_views:
            trace_ids += [int(traceView.attrib['traceDataRef'])]
            symbol.append(int(traceView.attrib['traceDataRef']))

        trace_ids = sorted(trace_ids)
        baseline_parsed[tuple(trace_ids)] = value
        hw.mapping[value] += [tuple(trace_ids)]
        segmentation.append(symbol)

    hw.baseline_parsed = baseline_parsed
    hw.symbol_stream = symbol_stream
    hw.segmentation = segmentation
    _flat_seg = [stroke2 for symbol2 in segmentation for stroke2 in symbol2]
    if len(_flat_seg) != len(recording):
        print "SEGMENTATION LENGTH IS OFF"
        return None
    assert len(_flat_seg) == len(recording), \
        ("Segmentation had length %i, but recording has %i strokes (%s)" %
         (len(_flat_seg), len(recording), filepath))
    assert set(_flat_seg) == set(range(len(_flat_seg)))
    hw.inkml = beautify_xml(filepath)
    hw.filepath = filepath
   # print "Segmentation: {}".format(hw.segmentation)
    for key, values in hw.mapping.iteritems():
        for val in values:
            hw.inv_mapping[val] = key

            # print "Before inverting: {}".format(hw.mapping)
            # print "After inverting: {}".format(hw.inv_mapping)
    return hw


def read_folder(folder, start, end):
    recordings = []
    parse_error = 0
    total_num_files = 0
    for directory in folder[start:end]:
        filenames = os.listdir(folder[0] + directory)
        invalid_inputs = 0
        for i, filename in enumerate(filenames):

            filename_copy = filename
            filename = folder[0] + directory + filename
            # print filename

            hw = read(folder, filename, filename_copy, directory)
            if hw == None:
                invalid_inputs += 1
                continue
            if hw.formula_in_latex is not None:
                hw.formula_in_latex = hw.formula_in_latex.strip()
            else:
                continue

            print hw.baseline_parsed
            recordings.append(hw)
            baseline_parsing = ""
            sorted_tuples = sorted(hw.baseline_parsed)
            for key in sorted_tuples:
                baseline_parsing += hw.baseline_parsed[key]

            perfect_parsing = "".join(hw.formula_in_latex.split())
            if perfect_parsing[0] == '$':
                baseline_parsing = "$" + baseline_parsing + "$"

            if perfect_parsing != baseline_parsing:
                parse_error += 1
            recordings.append(hw)
            total_num_files += 1
            # break


        print "INFO: Out of {} files, {} were not parsed properly. ".format(len(filenames), invalid_inputs)

    print "ACCURACY: Baseline parsing error: {}".format(1.0 * parse_error / total_num_files)

    TRAINING_Y = []
    TRAINING_X = []

    for index, hw in enumerate(recordings):
        x, y = hw.get_training_example()
        TRAINING_X += x
        TRAINING_Y += y

    TEST_X = []
    TEST_Y = []
    test_indices = random.sample(range(len(TRAINING_X)), len(TRAINING_X) / 10)
    for index in test_indices:
        TEST_X.append(TRAINING_X[index])
        TEST_Y.append(TRAINING_Y[index])

    TRAINING_X = [val for i, val in enumerate(TRAINING_X) if i not in test_indices]
    TRAINING_Y = [val for i, val in enumerate(TRAINING_Y) if i not in test_indices]

    X = TRAINING_X

    y_map = {}
    counter = 0
    NEW_TRAINING_Y = []
    weights = []
    freq = defaultdict(int)
    for y in TRAINING_Y:
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

    for example, y, new_y in zip(TRAINING_X, TRAINING_Y, NEW_TRAINING_Y):
        print len(example), y, new_y


    clf = svm.SVC(decision_function_shape='ovo', gamma=0.001, C=50.0)
    clf.fit(X, Y, weights)

    # hps, _, _ = optunity.maximize(svm_auc, num_evals=200, logC=[-5, 2], logGamma=[-5, 1])
    # clf = svm.SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma']).fit(X, Y)
    clf.decision_function_shape = "ovr"
    error = 0

    for i in range(len(TEST_X)):
        dec = clf.decision_function([TEST_X[i]])
        print "Dec: {}".format(dec)
        print "Max: {}".format(max(dec[0]))
        max_index = np.argmax(dec[0])
        for symbol, index in y_map.iteritems():
            if index == max_index:
                print "Matching symbol: {}, Truth: {}".format(symbol, TEST_Y[i])
                if symbol != TEST_Y[i]:
                    error += 1


    print "ACCURACY: SVM character recognition error: {}".format(1.0 * error / len(TEST_Y))

def main(folder,start,end):
    """
    Read folder.

    Parameters
    ----------
    folder : list of str
    """

    logging.info(folder)
    read_folder(folder,start,end)


def handler(signum, frame):
    """Add signal handler to safely quit program."""
    print('Signal handler called with signal %i' % signum)
    sys.exit(-1)

def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts


if __name__ == '__main__':
    TRAINING_X = []
    TRAINING_Y = []
    myargs = getopts(argv)
    start = int(myargs['-s'])
    end = int(myargs['-e'])
    signal.signal(signal.SIGINT, handler)
    folder = ["/Users/norahborus/Documents/latex-project/baseline/training_data/", "CROHME_training_2011/", "TrainINKML_2013/", "trainData_2012_part1/", "trainData_2012_part2/"]
    main(folder, start, end)

