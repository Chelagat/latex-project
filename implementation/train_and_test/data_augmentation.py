import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import collections
from load_images import add_to_test_v4
import util
def _elastic_transform_2D(images, sigma):
    """`images` is a numpy array of shape (K, M, N) of K images of size M*N."""
    # Take measurements

    rng = np.random.RandomState(42)
    alpha = np.random.randint(400,501)
    interpolation_order = 1
    image_shape = images[0].shape
    # Make random fields
    dx = rng.uniform(-1, 1, image_shape) * alpha
    dy = rng.uniform(-1, 1, image_shape) * alpha
    # Smooth dx and dy
    sdx = gaussian_filter(dx, sigma=sigma, mode='reflect')
    sdy = gaussian_filter(dy, sigma=sigma, mode='reflect')
    # Make meshgrid
    x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
    # Distort meshgrid indices
    distorted_indices = (y + sdy).reshape(-1, 1), \
                        (x + sdx).reshape(-1, 1)

    # Map cooordinates from image to distorted index set
    transformed_images = [map_coordinates(image, distorted_indices, mode='reflect',
                                          order=interpolation_order).reshape(image_shape)
                          for image in images]

    return transformed_images


import random



def test():
    filenames = ['/Users/norahborus/Desktop/example_phi.png']
    images = []
    for filename in filenames:
        image = Image.open(filename).convert('L')
        array = np.asarray(image)

        array = array / 255.0
        plt.imshow(array, cmap='gray')
        plt.savefig('example_phi.png')
        images.append(array)



    images = np.array(images)
    image_index = random.randint(0, len(images) - 1)
    randomized_sigma = 60 # randomize value of sigma
    batch = np.array(images[image_index]).reshape((1, 32, 32))
    result = _elastic_transform_2D(batch, randomized_sigma)[0]
    print result
    plt.imshow(result, cmap='gray')
    plt.savefig('result_phi.png')

def augment_symbol_data_for_nn(path, symbol, num_images_to_generate):
    print "Num images to generate: ", num_images_to_generate
    symbol_dir = path + symbol + '/'
    filenames = os.listdir(symbol_dir)
    images = []
    for filename in filenames:
        if 'DS_Store' in filename:
            continue
        if not os.path.isfile(symbol_dir + filename):
            continue
        image = Image.open(symbol_dir + filename).convert('L')
        array = np.asarray(image)
        array = array / 255.0
        images.append(array)


    if len(images) == 0:
        return None

    images = np.array(images)
    np.random.shuffle(images)

    sigma_range = range(40,41)

    for i in range(num_images_to_generate):
        image_index = random.randint(0, len(images)-1)
        randomized_sigma = sigma_range[random.randint(0, len(sigma_range)-1)]#randomize value of sigma
        batch = np.array(images[image_index]).reshape((1, 32,32))
        result = _elastic_transform_2D(batch, randomized_sigma)[0]
        plt.imshow(result, cmap='gray')
        augmented_dir = path + 'AUGMENTED/' + symbol + '/'
        if not os.path.exists(augmented_dir):
            os.makedirs(augmented_dir)
        plt.savefig("{}sigma={}_{}".format(augmented_dir, randomized_sigma, filenames[image_index]))
        print "Generating images for symbol ", symbol, " --> Done with image #", i

def augment_symbol_data_for_svm(path, symbol, num_images_to_generate):
    print "Num images to generate: ", num_images_to_generate
    symbol_dir = path + symbol + '/'
    filenames = os.listdir(symbol_dir)
    images = []
    for filename in filenames:
        if 'DS_Store' in filename:
            continue
        if not os.path.isfile(symbol_dir + filename):
            continue
        image = Image.open(symbol_dir + filename).convert('L')
        array = np.asarray(image)
        array = array / 255.0
        images.append(array)


    if len(images) == 0:
        return None

    images = np.array(images)
    np.random.shuffle(images)

    sigma_range = range(40,61)

    for i in range(num_images_to_generate):
        image_index = random.randint(0, len(images)-1)
        randomized_sigma = sigma_range[random.randint(0, len(sigma_range)-1)]#randomize value of sigma
        batch = np.array(images[image_index]).reshape((1, 200,200))
        result = _elastic_transform_2D(batch, randomized_sigma)[0]
        plt.imshow(result, cmap='gray')
        augmented_dir = path + 'AUGMENTED/' + symbol + '/'
        if not os.path.exists(augmented_dir):
            os.makedirs(augmented_dir)
        plt.savefig("{}sigma={}_{}".format(augmented_dir, randomized_sigma, filenames[image_index]))
        print "Generating images for symbol ", symbol, " --> Done with image #", i


import cPickle



def combined_with_augmentation_limit_500(svm=True):
    symbols = ['!', '(', ')', '+', ',', '-', 'dot', 'forward_slash', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'B', 'C', 'E',
     'F', 'G', 'H', 'I', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Y', '[', '\\Delta', '\\alpha', '\\beta', '\\cos',
     '\\div', '\\exists', '\\forall', '\\gamma', '\\geq', '\\gt', '\\in', '\\infty', '\\int', '\\lambda', '\\ldots',
     '\\leq', '\\lim', '\\log', '\\lt', '\\mu', '\\neq', '\\phi', '\\pi', '\\pm', '\\prime', '\\rightarrow', '\\sigma',
     '\\sin', '\\sqrt', '\\sum', '\\tan', '\\theta', '\\times', '\\{', '\\}', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|']
    if svm:
        from_path = '/Users/norahborus/Documents/DATA/CLASSES/200x200/'
    else:
        from_path = '/Users/norahborus/Documents/DATA/CLASSES/32x32/'

    all_images = []
    for symbol in symbols:
        orig_data_dir = from_path + symbol + '/'
        augmented_dir = from_path + 'AUGMENTED/' + symbol + '/'
        augmented_images = os.listdir(augmented_dir)
        orig_images = os.listdir(orig_data_dir)
        augmented_images_full_path = [augmented_dir + filename for filename in augmented_images]
        orig_images_full_path = [orig_data_dir + filename for filename in orig_images]
        combined = np.array(augmented_images_full_path + orig_images_full_path)
        np.random.shuffle(combined)
        random_sample = combined[:500]
        all_images += random_sample

    print "Size of all Images: {}".format(len(all_images))

    return all_images


def read_images_nn_augmented():
    TRAINING_X = []
    TRAINING_Y = []
    TEST_X = []
    TEST_Y = []
    y_map = {}
    counter = 0
    filenames = combined_with_augmentation_limit_500(svm=False)
    train_y_freq = defaultdict(int)
    test_y_freq = defaultdict(int)
    for index, filename in enumerate(filenames):
        x = filename
        if 'kml' not in filename:
            continue

        if 'png' not in filename:
            y = filename[filename.index('kml')+3:]
            if  y == 'forward_slash':
                y = '/'
        else:
            y = filename[filename.index('kml')+3:-4]
            if  y == 'forward_slash':
                y = '/'

        if y not in y_map:
            y_map[y] = counter
            counter += 1

        if add_to_test_v4():
            test_y_freq[y] += 1
            TEST_X.append(x)
            TEST_Y.append(y)
        else:
            train_y_freq[y] += 1
            TRAINING_X.append(x)
            TRAINING_Y.append(y)

    print len(TRAINING_Y), len(TEST_Y)
    return TRAINING_X, TRAINING_Y, TEST_X, TEST_Y, y_map

def read_images_svm_augmented():
    TRAINING_X = []
    TRAINING_Y = []
    TEST_X = []
    TEST_Y = []
    y_map = {}
    counter = 0

    train_filenames = combined_with_augmentation_limit_500(svm=True)
    train_y_freq = defaultdict(int)
    test_y_freq = defaultdict(int)

  #  np.random.shuffle(train_filenames)
  #  np.random.shuffle(test_filenames)

    print util.commonly_missegmented_symbols

    for index, filename in enumerate(train_filenames):
        print "Train file: {}".format(index)
        if 'kml' not in filename:
          continue
        x = filename
        image = Image.open(x).convert('L')
        array = (np.asarray(image)).flatten()

        if 'png' not in filename:
            y = filename[filename.index('kml')+3:]
            if  y == 'forward_slash':
                y = '/'
        else:
            y = filename[filename.index('kml')+3:-4]
            if  y == 'forward_slash':
                y = '/'
        if y not in y_map:
            y_map[y] = counter
            counter += 1

        if y in util.commonly_missegmented_symbols:
            if add_to_test_v4():
                test_y_freq[y] += 1
                TEST_X.append(array)
                TEST_Y.append(y)
            else:
                train_y_freq[y] += 1
                TRAINING_X.append(array)
                TRAINING_Y.append(y)
        else:
            TRAINING_X.append(array)
            TRAINING_Y.append(y)


    print len(TRAINING_Y), len(TEST_Y)
    return TRAINING_X, TRAINING_Y, TEST_X, TEST_Y, y_map


if __name__ == '__main__':



    svm_path = '/Users/norahborus/Documents/DATA/CLASSES/200x200/'
    nn_path = '/Users/norahborus/Documents/DATA/CLASSES/32x32_Test/'
    '''
    symbols = ['!', '(', ')', '+', ',', '-', 'dot', 'forward_slash', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'B', 'C', 'E',
     'F', 'G', 'H', 'I', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Y', '[', '\\Delta', '\\alpha', '\\beta', '\\cos',
     '\\div', '\\exists', '\\forall', '\\gamma', '\\geq', '\\gt', '\\in', '\\infty', '\\int', '\\lambda', '\\ldots',
     '\\leq', '\\lim', '\\log', '\\lt', '\\mu', '\\neq', '\\phi', '\\pi', '\\pm', '\\prime', '\\rightarrow', '\\sigma',
     '\\sin', '\\sqrt', '\\sum', '\\tan', '\\theta', '\\times', '\\{', '\\}', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|']
    '''
    rare_symbols = {'\\Delta', '\\exists', '\\forall', '\\gamma', '\\geq', '\\gt', '\\in', '\\lambda', '\\ldots',
                    '\\leq', '\\lim', '\\log', '\\lt', '\\mu', '\\neq', '\\phi', '\\pm', '\\prime', '\\rightarrow',
                    '\\sigma',
                    }
    with open('DATA_ANALYSIS/aggregate_class_distribution', 'rb')  as fp:
        freq = cPickle.load(fp)

    freq['dot'] = freq['.']
    freq['forward_slash'] = freq['/']
  #  augment_symbol_data(path, symbol, 500 - freq[symbol])

    '''
    for index, symbol in enumerate(rare_symbols):
        if freq[symbol] > 200:
            continue

        augment_symbol_data_for_svm(svm_path, symbol, 200-freq[symbol])
        print "DONE with symbol #{}, -->{} ".format(index, symbol)
    '''

    for index, symbol in enumerate(rare_symbols):
        if freq[symbol] > 200:
            continue
        augment_symbol_data_for_nn(nn_path, symbol, 200-freq[symbol])
        print "DONE with symbol #{}, -->{} ".format(index, symbol)

