import sys, os, random
from datasets.inkml import read
from ast import literal_eval
import numpy as np
from numpy import array_equal

def read_data(files):
	total_correct, total, total_correct_chars, total_chars = 0, 0, 0, 0
	for file in files:
		hw = read('../baseline/training_data/CHROME_training_2011', '../baseline/training_data/CHROME_training_2011/' + file, file, 'CHROME_training_2011')
		traces = literal_eval(hw.raw_data_json)
		
		# Get trace bounds

		trace_bounds = []
		for t in traces:
			max_x, min_x, max_y, min_y = float('-inf'), float('+inf'), float('-inf'), float('+inf')
			for d in t:
				y = d['y']
				x = d['x']
				if y > max_y: max_y = y
				elif y < min_y: min_y = y
				if x > max_x: max_x = x
				elif x < min_x: min_x = x
			trace_bounds.append((max_x,min_x,max_y,min_y))

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

def main():
	files = os.listdir('../baseline/training_data/CHROME_training_2011')
	total_correct, total, total_correct_chars, total_chars = read_data(files)
	print 'FIRST ACCURACY: ', 100.*total_correct / total
	print 'SECOND ACCURACY: ', 100.*total_correct_chars / total_chars

if __name__ == '__main__':
	main()