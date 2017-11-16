import numpy as np
import pickle
import itertools
x = np.zeros((3,1))
x_list = []
for i in xrange(5):
    x = np.zeros((3,1))
    flattened = list(itertools.chain.from_iterable(x))
    x_list.append(x)

print "X before:{}, {}, {}".format(type(x_list), len(x_list), x_list)
with open("pickle-test-file.pickle", "w") as fp:
    pickle.dump(x_list, fp)



with open("pickle-test-file.pickle", "rb") as fp:
    x_list = pickle.load(fp)


print "X after: {}, {}, {}".format(type(x_list), len(x_list), x_list)