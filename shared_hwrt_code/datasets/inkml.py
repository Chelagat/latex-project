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
    #  print short_filename
    """
    Read a single InkML file

    Parameters
    ----------
    filepath : string
        path to the (readable) InkML file

    Returns
    -------
    HandwrittenData :
        The parsed InkML file as a HandwrittenData object
    """
    import xml.etree.ElementTree as ET
    data = ""
    with open(filepath, "r") as myfile:
        data = myfile.read()

    myfile.close()
    try:
        root = ET.fromstring(data)
    except:
        return None

    if root == None:
        return None
   
   # import xml.etree.ElementTree
   # root = xml.etree.ElementTree.parse(filepath).getroot()
   # print root
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
    for tg in trace_group.findall('{http://www.w3.org/2003/InkML}traceGroup'):
        annotations = tg.findall('{http://www.w3.org/2003/InkML}annotation')
        # anno_xml = tg.findall('{http://www.w3.org/2003/InkML}annotationXML')
        if len(annotations) != 1:
            raise ValueError("%i annotations found for '%s'." %
                             (len(annotations), filepath))

        value = annotations[0].text
        '''
        db_id = formula_to_dbid(normalize_symbol_name(annotations[0].text))
        symbol_stream.append(db_id)
        '''

        '''
        Need some sort of mapping from symbol to strokes
        '''

        trace_views = tg.findall('{http://www.w3.org/2003/InkML}traceView')
        symbol = []
        trace_ids = []
        for traceView in trace_views:
            trace_ids += [int(traceView.attrib['traceDataRef'])]
            symbol.append(int(traceView.attrib['traceDataRef']))

        hw.mapping[value] += [tuple(trace_ids)]
        segmentation.append(symbol)
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
    global TRAINING_X
    global TRAINING_Y

    """
    Parameters
    ----------
    folder : string
        Path to a folde with *.inkml files.

    Returns
    -------
    list :
        Objects of the type HandwrittenData
    """
    import glob
    recordings = []

    X_NP_FOLDER = "/afs/.ir/users/n/b/nborus/latex_project/latex-project/shared_hwrt_code/datasets/saved_np_arrays_directory_"+str(start)+"/"
    LATEX_TRUTH_FILE = "/afs/.ir/users/n/b/nborus/latex_project/latex-project/shared_hwrt_code/datasets/saved_latex_truth_directory_"+str(start)+".txt"
    SVM_MODEL_FILE = "/afs/.ir/users/n/b/nborus/latex_project/latex-project/shared_hwrt_code/datasets/svm_model_directory_"+str(start)+".pickle"
    for directory in folder[start:end]:
        filenames = os.listdir(folder[0] + directory)
        error = 0
       # print "FILENAMES: {}".format(filenames)
        for i, filename in enumerate(filenames):  # natsorted(glob.glob("%s/*.inkml" % folder)):

            # filename = "formulaire001-equation003.inkml"
            filename_copy = filename
            filename = folder[0] + directory + filename
            # print filename

            hw = read(folder, filename, filename_copy, directory)
	   # print hw.formula_in_latex
            if hw == None:
                error += 1
                continue
            if hw.formula_in_latex is not None:
                hw.formula_in_latex = hw.formula_in_latex.strip()
            #if hw.formula_in_latex is None or \
            #    not hw.formula_in_latex.startswith('$') or \
            #    not hw.formula_in_latex.endswith('$'):
            #    continue
            '''
            if hw.formula_in_latex is not None:
                logging.info("Starts with: %s",
                             str(hw.formula_in_latex.startswith('$')))
                logging.info("ends with: %s",
                             str(hw.formula_in_latex.endswith('$')))
            logging.info(hw.formula_in_latex)
            logging.info(hw.segmentation)
            hw.show()
            '''

            print hw.formula_in_latex
            recordings.append(hw)
            # break

        print "Out of {} files, {} were not parsed properly. ".format(len(filenames), error)

    TRAINING_Y = []
    TRAINING_X = []

    for index, hw in enumerate(recordings):
        x, y = hw.get_training_example()
	print "Done with one equation: #{}".format(index)
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

    print "Done"
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

    '''
    x_test = X
    y_test = Y

    def svm_auc(logC, logGamma):
        model = svm.SVC(C=10 ** logC, gamma=10 ** logGamma).fit(X, Y)
        decision_values = model.decision_function(x_test)
        return optunity.metrics.roc_auc(y_test, decision_values)

    '''

    clf = svm.SVC(decision_function_shape='ovo', gamma=0.001, C=50.0)
    clf.fit(X, Y, weights)

    with open(SVM_MODEL_FILE, "w") as fp:
        pickle.dump(clf, fp)

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

        '''
        data = np.reshape(X[i], (480,640))
        pl.imshow(data)
        pl.savefig("results/Result_{}".format(i))
        '''

    print "error: {}".format(1.0 * error / len(TEST_Y))
    '''
    latex_truth = []
    for hw in recordings:
        x,y =  hw.get_training_example()
        filename = X_NP_FOLDER + hw.filename[:hw.filename.index('.')] + ".pickle"
        with open(filename, "w") as fp:  # Pickling
            pickle.dump(x, fp)

        latex_truth += y

    with open(LATEX_TRUTH_FILE, "w") as fp:
        pickle.dump(latex_truth, fp)

    print "Done with saving to file: Saved: {} training examples.".format(len(recordings))
    '''

def svm_train_test(start,end):
    X_NP_FOLDER = "/afs/.ir/users/n/b/nborus/latex_project/latex-project/shared_hwrt_code/datasets/saved_np_arrays_directory_"+str(start)+"/"
    LATEX_TRUTH_FILE = "/afs/.ir/users/n/b/nborus/latex_project/latex-project/shared_hwrt_code/datasets/saved_latex_truth_directory_"+str(start)+".txt"
    SVM_MODEL_FILE = "/afs/.ir/users/n/b/nborus/latex_project/latex-project/shared_hwrt_code/datasets/svm_model_directory_"+str(start)+".pickle"
    with open(SVM_MODEL_FILE, "r") as fp:
	clf = pickle.load(fp)

    print type(clf)
    TRAINING_Y = []
    TRAINING_X = []

    with open(LATEX_TRUTH_FILE, "rb") as fp:
        TRAINING_Y = pickle.load(fp)

    x_files = os.listdir(X_NP_FOLDER)
    for x_file in x_files:
        x_file = X_NP_FOLDER + x_file
        with open(x_file, "rb") as fp:
            TRAINING_X += pickle.load(fp)

    for x,y in zip(TRAINING_X, TRAINING_Y):
	print "x:{},  y: {}".format(x,y)

def svm_train():
    X_NP_FOLDER = "/afs/.ir/users/n/b/nborus/latex_project/latex-project/shared_hwrt_code/datasets/saved_np_arrays/"
    LATEX_TRUTH_FILE = "/afs/.ir/users/n/b/nborus/latex_project/latex-project/shared_hwrt_code/datasets/saved_latex_truth.txt"
    SVM_MODEL_FILE = "/afs/.ir/users/n/b/nborus/latex_project/latex-project/shared_hwrt_code/datasets/svm_model.pickle"
    TRAINING_Y = []
    TRAINING_X = []

    with open(LATEX_TRUTH_FILE, "rb") as fp:
        TRAINING_Y = pickle.load(fp)

    x_files = os.listdir(X_NP_FOLDER)
    for x_file in x_files:
        x_file = X_NP_FOLDER + x_file
        with open(x_file, "rb") as fp:
            TRAINING_X += pickle.load(fp)

    TEST_X = []
    TEST_Y = []
    test_indices = random.sample(range(len(TRAINING_X)), len(TRAINING_X) / 10)
    for index in test_indices:
        TEST_X.append(TRAINING_X[index])
        TEST_Y.append(TRAINING_Y[index])

    TRAINING_X = [val for i, val in enumerate(TRAINING_X) if i not in test_indices]
    TRAINING_Y = [val for i, val in enumerate(TRAINING_Y) if i not in test_indices]

    print "Done"
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

    '''
    x_test = X
    y_test = Y

    def svm_auc(logC, logGamma):
        model = svm.SVC(C=10 ** logC, gamma=10 ** logGamma).fit(X, Y)
        decision_values = model.decision_function(x_test)
        return optunity.metrics.roc_auc(y_test, decision_values)

    '''

    clf = svm.SVC(decision_function_shape='ovo', gamma=0.100, C=1000.0)
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

        '''
        data = np.reshape(X[i], (480,640))
        pl.imshow(data)
        pl.savefig("results/Result_{}".format(i))
        '''

    print "error: {}".format(1.0 * error / len(TEST_Y))


def main(folder,start,end):
    """
    Read folder.

    Parameters
    ----------
    folder : str
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
    folder = ["/afs/.ir/users/n/b/nborus/latex_project/latex-project/baseline/training_data/", "CHROME_training_2011/", "TrainINKML_2013/", "trainData_2012_part1/", "trainData_2012_part2/"]
    main(folder, start, end)
   # svm_train_test(start,end)

