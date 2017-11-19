import sys, os, random
from inkml import read
from ast import literal_eval
import numpy as np
from numpy import array_equal


def segmentation_overlap_heuristic(path, folders):
    total_correct, total, total_correct_chars, total_chars = 0, 0, 0, 0

    for folder in folders:
        print "Folder: {}".format(folder)
        files = os.listdir(path+folder)
        for file in files:
            hw = read(path + folder, path+folder+file, file,folder)
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

            for g in seg_guess:
                if g in hw.segmentation: total_correct_chars += 1

            total_chars += len(hw.segmentation)

            if array_equal(np.array(seg_guess), np.array(hw.segmentation)):
                total_correct += 1

            total += 1

    return total_correct, total, total_correct_chars, total_chars


def segmentation_baseline(path, folders):
    total_correct, total, total_correct_chars, total_chars = 0, 0, 0, 0
    for folder in folders:
        print "Folder: {}".format(folder)
        files = os.listdir(path+folder)
        for file in files:
            hw = read(path + folder, path+folder+file, file,folder)
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




def main():
    training_folders = ["CHROME_training_2011/", "TrainINKML_2013/", "trainData_2012_part1/", "trainData_2012_part2/"]
    total_correct_1, total_1, total_correct_chars_1, total_chars_1 = segmentation_baseline(
        '/Users/norahborus/Documents/latex-project/baseline/training_data/',training_folders)

    total_correct_2, total_2, total_correct_chars_2, total_chars_2 = segmentation_overlap_heuristic(
        '/Users/norahborus/Documents/latex-project/baseline/training_data/', training_folders)

    print "*****************************************************************"
    print 'BASELINE FIRST ACCURACY --> ', 100. * total_correct_1 / total_1
    print 'BASELINE: SECOND ACCURACY --> ', 100. * total_correct_chars_1 / total_chars_1
    print "*****************************************************************"
    print "*****************************************************************"
    print 'OVERLAP HEURISTIC: FIRST ACCURACY --> ', 100. * total_correct_2 / total_2
    print 'OVERLAP HEURISTIC: SECOND ACCURACY --> ', 100. * total_correct_chars_2 / total_chars_2
    print "*****************************************************************"



if __name__ == '__main__':
    main()
