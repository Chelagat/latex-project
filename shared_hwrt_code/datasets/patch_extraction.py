import numpy as np
from sklearn.feature_extraction import image
from PIL import Image, ImageChops
from skimage import data, color, exposure

im = Image.open('/Users/norahborus/Downloads/satire.jpg')
np_image = color.rgb2gray(np.asarray(im))
print np_image.shape
patches = image.extract_patches_2d(np_image, (2, 2), max_patches=10, random_state=0)
print patches.shape
print patches