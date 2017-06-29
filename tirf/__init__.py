import numpy as np
from PIL import Image

from . import preprocessing
from . import hog
from . import sift
from . import main

__version__ = 1.0

def visualize_sift(path, *, n=100):
    img = np.asarray(Image.open(path))[:, :, :3].copy()
    img = preprocessing.grayscale(img)
    return sift.visualize_sift_descriptors(img)


def visualize_hog(path):
    feats = main.get_hog_features(path)

    img = np.asarray(Image.open(path))[:, :, :3].copy()
    img = preprocessing.grayscale(img)
    img = preprocessing.bilinear_resize(img, 64, 128)

    return main.visualize_hog(feats, img)
