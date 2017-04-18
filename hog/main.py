import time

from PIL import Image
import numpy

from . import preprocessing
from . import hog


def logging_begin(msg):
    print('## HOG: {:<10}...'.format(msg))


def logging_end():
    print('----->  OK')


def detect_human(img_path):
    begin_time = time.time()

    logging_begin('Loading {}'.format(img_path))
    img = numpy.asarray(Image.open(img_path)).copy()
    logging_end()

    logging_begin('Preprocessing')
    img = preprocessing.grayscale(img)
    img = preprocessing.bilinear_resize(img, 64, 128)
    logging_end()

    logging_begin('Computing hog')
    gradient_array = preprocessing.compute_gradient(img)
    magnitude_array = preprocessing.get_magnitude(img, gradient_array)
    direction_array = preprocessing.get_direction(img, gradient_array)
    img_histogram = hog.compute_all_histograms(magnitude_array, direction_array, 8)
    img_blocks = hog.create_blocks(img_histogram, 2, 8, img.shape[0], img.shape[1])
    hog_feature_vector = hog.concatenate_blocks(img_blocks)
    logging_end()

    print(hog_feature_vector)

    print('DONE in {}s'.format(str(round(time.time() - begin_time, 2))))
    img = Image.fromarray(img)
    img.show()


