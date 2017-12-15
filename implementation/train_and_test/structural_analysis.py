import sys, os, random
from datasets.inkml import read
from ast import literal_eval
import numpy as np
from numpy import array_equal
import collections
from enum import Enum

class Rel(Enum):
		SUB = 1
		SUP = 2
		ENC = 3
		NEX = 4
		CON = 5


class SA:
	def __init__(self):
		pass

	def group(self, inv_map, seg, traces):

		# Find trace bounds:
		trace_bounds = []
		for i, s in enumerate(seg):
			max_x, min_x = float('-inf'), float('+inf')
			max_y, min_y = float('-inf'), float('+inf')
			for j in s:
				t = traces[i]
				for d in t:
					x = d['x']
					y = d['y']
					if x > max_x: max_x = x
					if x < min_x: min_x = x
					if y > max_y: max_y = y
					if y < min_y: min_y = y
			trace_bounds.append((i,(max_x,min_x,max_y,min_y)))

		# Sort trace bounds by tb[1]
		trace_bounds = sorted(trace_bounds, key=lambda tb: tb[1][1])
		ordered_seg = [ seg[i] for i, t in trace_bounds ]

		relationship_dict = {}

		for i, tb1 in enumerate(trace_bounds):
			for j in range(len(trace_bounds)):
				if j <= i: continue
				tb2 = trace_bounds[j]
				# Check for complete containment by width
				if tb1[0] >= tb2[0] and tb1[1] <= tb2[1]:
					relationship_dict[(tuple(seg[i]),tuple(seg[j]))] = Rel.ENC
				# Check for super script by x and y heuristic
				elif abs(tb1[0] - tb2[1]) < 1 and ((tb1[2] + tb1[3])/2.) < tb2[3]:# and (abs(tb2[2] - tb2[3])) < 0.75*(abs(tb1[2] - tb1[3])):
					relationship_dict[(tuple(seg[i]),tuple(seg[j]))] = Rel.SUP
				# Check for sub script by x and y heuristic
				elif abs(tb1[0] - tb2[1]) < 1 and ((tb1[2] + tb1[3])/2.) > tb2[2]:# and (abs(tb2[2] - tb2[3])) < 0.75*(abs(tb1[2] - tb1[3])):
					relationship_dict[(tuple(seg[i]),tuple(seg[j]))] = Rel.SUB
				# Check for connected
				elif abs(tb1[0] - tb2[1]) < 0.1:
					relationship_dict[(tuple(seg[i]),tuple(seg[j]))] = Rel.CON
				# Check for next
				else:
					relationship_dict[(tuple(seg[i]),tuple(seg[j]))] = Rel.NEX
		return relationship_dict

	def classify(self, test):
		# ordered_groups = self.group(test[0], test[1], test[2])
		relationships = self.group(test[0], test[1], test[2])
		return relationships
		# return (ordered_groups, self.relationships)
		# exp = self.convert(test[0], ordered_groups, test[2])
		# return '$' + exp.reconstruct() + '$'

def rel_to_string(rel):
	if rel == Rel.SUB:
		return 'SUB'
	elif rel == Rel.SUP:
		return 'SUP'
	elif rel == Rel.ENC:
		return 'ENC'
	elif rel == Rel.NEX:
		return 'NEX'
	elif rel == Rel.CON:
		return 'CON'
	else:
		return 'NON'

def test(SA, test_input, test_output):
	correct, chars_correct, total_chars = 0, 0, 0
	for i, (test, truth) in enumerate(zip(test_input, test_output)):
		if i == 0: continue
		relationships = SA.classify(test)
		print 'LATEX: ', truth
		print 'RELS: '
		for k, rel in relationships.iteritems():
			print '(' + test[0][k[0]] + ', ' + test[0][k[1]] + ')'  + ' ---> ' + rel_to_string(rel)
	
def create_tests(_input, _output, _files, directory, parent):
	for file in _files:
		hw = read(directory, directory + file, file, parent)
		traces = literal_eval(hw.raw_data_json)

		# For now, assume perfect CSeg and OCR
		_input.append([hw.inv_mapping, hw.segmentation, traces])
		_output.append(hw.formula_in_latex)

def split_data(files):
	random.seed(123)
	test_files = []
	for i in range(len(files)/10):
		test_files.append(random.choice(files))
		files.remove(test_files[i])
	return files, test_files

def main():
	directories = ['../baseline/training_data/CHROME_training_2011/']
	parents = ['CHROME_training_2011']

	training_input, training_output = [], []
	test_input, test_output = [], []

	for i in range(len(directories)):
		files = os.listdir(directories[i])
		training_files, test_files = split_data(files)
		create_tests(test_input, test_output, test_files, directories[i], parents[i])

	#### TEST ####
	## BY HAND? ##
	test(SA(), test_input, test_output)

if __name__ == '__main__':
	main()