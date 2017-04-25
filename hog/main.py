import time

from PIL import Image
import numpy as np

from . import preprocessing
from . import hog


def logging_begin(msg, verbose):
    if verbose:
        print('## HOG: {:<10}...'.format(msg))


def logging_end(verbose):
    if verbose:
        print('----->  OK')


def get_hog_features(img_path, verbose=False):
    begin_time = time.time()

    logging_begin('Loading {}'.format(img_path), verbose)
    img = np.asarray(Image.open(img_path))[:,:,:3].copy()
    logging_end(verbose)

    logging_begin('Preprocessing', verbose)
    img = preprocessing.grayscale(img)
    img = preprocessing.bilinear_resize(img, 64, 128)
    logging_end(verbose)

    logging_begin('Computing hog', verbose)
    gradient_array = preprocessing.compute_gradient(img)
    magnitude_array = preprocessing.get_magnitude(img, gradient_array)
    direction_array = preprocessing.get_direction(img, gradient_array)
    img_histogram = hog.compute_all_histograms(magnitude_array, direction_array, 8)
    img_blocks = hog.create_blocks(img_histogram, 2, 8, img.shape[0], img.shape[1])
    hog_feature_vector = hog.concatenate_blocks(img_blocks)
    logging_end(verbose)

    logging_begin('DONE in {}s'.format(str(round(time.time() - begin_time, 3))),
                  verbose)

    return hog_feature_vector

