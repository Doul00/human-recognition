import math

import numpy as np
from scipy import misc
from scipy.ndimage import filters


def create_sift_features(img):
    """
    See: http://docs.opencv.org/trunk/da/df5/tutorial_py_sift_intro.html
    """
    print(harris_score(img))

def compute_octaves(img, *, sigma=1.6, octave_nb=5):
    """
    Computes @octave_nb octaves with a @sigma starting from 1.6 and growing
    each time by a factor a sqrt(2).
    """
    octaves = []

    for _ in range(octave_nb):
        octaves.append(filters.gaussian_filter(img, sigma))
        sigma *= math.sqrt(2)

    return octaves


def scale_down(img):
    """
    Scales down the @img to half its original size.
    """
    return misc.imresize(img, .5)


def difference_of_gaussian(img):
    """
    Computes the Difference Of Gaussian:
        Several octaves are made, each a gaussian blur of the precedent with
        sigma starting from 1.6 and growing up by a factor of sqrt(2).
        Local extrema (max or min) are selected as potential keypoints between
        several layers of octaves.
    """
    octaves = compute_octaves(img)
    mid_octave = len(octaves) // 2
    keypoints = [] # Are made of tuples of coordinates (x, y)

    rows    = img.shape[0]
    columns = img.shape[1]
    print(rows * columns)
    print(rows, columns)
    for y in range(1, rows-1): # We are skipping the border pixels as they
        for x in range(1, columns-1): # probably won't be keypoints.
            min_extrema = float('inf')
            max_extrema = float('-inf')

            potential_keypoint = octaves[mid_octave].item(y, x, 0)

            for yy in range(-1, 2):
                for xx in range(-1, 2):
                    coord_y = y + yy
                    coord_x = x + xx
                    for octave in octaves:
                        min_extrema = min(min_extrema,
                                          octave.item(coord_y, coord_x, 0))
                        max_extrema = max(max_extrema,
                                          octave.item(coord_y, coord_x, 0))


            if potential_keypoint >= max_extrema or\
               potential_keypoint <= min_extrema:
                keypoints.append((x, y))


    print(len(keypoints))


def harris_score(img, *, sigma=1.6):
    """
    Computes the harris score (from the harris corner detection algorithm)
    for each pixel.
    See for all the related notation:
        http://aishack.in/tutorials/harris-corner-detector
    """
    # First we are computing the derivating for each axis (x, and y) found
    # with the Taylor series.
    i_x = filters.gaussian_filter(img, sigma=sigma, order=0)
    i_y = filters.gaussian_filter(img, (sigma,sigma), (1,0))

    i_xx = filters.gaussian_filter(i_x * i_x, sigma)
    i_yy = filters.gaussian_filter(i_y * i_y, sigma)
    i_xy = filters.gaussian_filter(i_x * i_y, sigma)

    # We are now computing the score R for a window with the eigenvalues of
    # the matrix:
    #   M = [ Ix^2  IxIy ]
    #       [ IxIy  Iy^2 ]
    # With R = det M - trace(M)^2

    det_M = i_xx * i_yy - i_xy ** 2
    tr_M  = i_xx + i_yy

    return det_m - tr_m ** 2

