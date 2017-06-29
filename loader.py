import glob
import sys
import os
import pickle

import numpy as np
from scipy.misc import imread
from PIL import Image

import tirf


def get_N_images(folder, folder_suffix, phase='train', validity='pos',
                 N=None, verbose=False, algo=None):
    """
    Loads N images from the selected 'phase' with the choosen 'validity'.
    If N is None, the function will load all existing images.
    """
    path_regex = os.path.join(folder, phase + folder_suffix, validity, '*.png')
    array = []

    for n, img_path in enumerate(glob.iglob(path_regex)):
        if N is not None and n >= N:
            return array
        if verbose and n % 100 == 0:
            print('type {}: {}th image...'.format(validity, n))
        array.append(get_image(img_path, algo))

    if verbose:
        print('Finished {} - {}'.format(phase, validity))

    return array


def get_image(img_path, algo):
    """
    Return HOG features of the image from 'img_path'.
    """
    if algo == 'hog':
        return tirf.main.get_hog_features(img_path)
    elif algo == 'sift':
        img = np.asarray(Image.open(img_path))[:,:,:3].copy()
        img = tirf.preprocessing.grayscale(img)
        img = tirf.preprocessing.bilinear_resize(img, 128, 256)
        return tirf.sift.get_sift_descriptors(img)
    else:
        img = np.asarray(Image.open(img_path))[:,:,:3].copy()
        img = tirf.preprocessing.grayscale(img)
        img = tirf.preprocessing.bilinear_resize(img, 64, 128)
        img = img[:,:,0]
        img = np.reshape(img, (img.shape[0] * img.shape[1]))
        return img


def get_N_targets(validity='pos', N=100):
    """
    Return an 1d-array for the y-target, either 1 for a human positivity, else 0.
    """
    if validity == 'pos':
        return np.ones((N, 1))
    return np.zeros((N, 1))


def get_set(folder, folder_suffix, phase='train', N=None, algo=None):
    """
    Return two arrays, one for input, one for target.
    """
    print('Loading in {}/{}{}'.format(folder, phase, folder_suffix))

    set_x_pos = get_N_images(folder, folder_suffix, phase=phase, validity='pos',
                             verbose=True, N=N, algo=algo)
    set_x_neg = get_N_images(folder, folder_suffix, phase=phase, validity='neg',
                             verbose=True, N=N, algo=algo)

    set_x = np.array(set_x_pos + set_x_neg).astype('float32')

    set_y = np.concatenate((get_N_targets(validity='pos', N=len(set_x_pos)),
                          get_N_targets(validity='neg', N=len(set_x_neg))))

    return set_x, set_y


def test_accuracy(clf, test_x, test_y):
    predictions = clf.predict(test_x)
    return sum(1 if real == prediction else 0
               for real, prediction in zip(test_y, predictions)) / len(test_x)


def save_classifier(clf, *, name):
    s = pickle.dumps(clf)
    with open(name, 'wb+') as f:
        f.write(s)


def load_classifier(*, name):
    with open(name, 'rb') as f:
        return pickle.loads(f.read())
