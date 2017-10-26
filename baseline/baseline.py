import collections
from os import listdir
from os.path import isfile, join

def updateMarkdown(markdown, filename):
	with  open(filename) as f:
		for line in f:
			if line.find('<mo') == -1 and line.find('<mi') == -1: continue
			line = line[:-2].lstrip()
			start = line.find('>')
			end = line.find('<',1)
			markdown[line[start+1:end]] += 1


def train(paths):
	markdown = collections.defaultdict(int)

	for d in paths:
		for f in listdir(d):
			if isfile(join(d,f)):
				updateMarkdown(markdown, d + '/' + f)
	markdown = [ (v, m) for m, v in markdown.iteritems() ]
	markdown.sort(reverse=True)
	return markdown[0][1]

def guess(guessingCharacter, filename):
	correct, total = 0, 0
	with open(filename) as f:
		for line in f:
			if line.find('<mo') == -1 and line.find('<mi') == -1: continue
			total += 1
			line = line[:-2].lstrip()
			start = line.find('>')
			end = line.find('<',1)
			if line[start+1:end] == guessingCharacter: correct += 1
	return correct, total


def testTrace(prediction, filename):
	with open(filename) as f:
		for line in f:



def test(guessingCharacter, d):
	correct, total = 0, 0
	for f in listdir(d):
		if isfile(join(d,f)):
			c, t = guess(guessingCharacter, d + '/' + f)
			correct += c
			total += t
	return float(correct)/total

def crossValidation(paths):
	accuracy = 0
	for i in range(len(paths)):
		training_paths = paths[:i] + paths[i+1:]
		guessingCharacter = train(training_paths)
		accuracy += test(guessingCharacter, paths[i])
	return accuracy/len(paths)

def main():
	paths = ['./training_data/CHROME_training_2011', 
			'./training_data/MatricesTrain2014',
			 './training_data/trainData_2012_part1', 
			 './training_data/trainData_2012_part2', 
			 './training_data/TrainINKML_2013']
	accuracy = crossValidation(paths)
	print 'ACCURACY: ', accuracy




if __name__ == '__main__':
	main()
