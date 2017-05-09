import math

import numpy as np
from scipy import misc
from scipy.ndimage import filters

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
                        print('t: ', octave.item(coord_y, coord_x, 0))
                        min_extrema = min(min_extrema,
                                          octave.item(coord_y, coord_x, 0))
                        max_extrema = min(max_extrema,
                                          octave.item(coord_y, coord_x, 0))

                        print(min_extrema, max_extrema)
            exit()
            if potential_keypoint >= max_extrema or\
               potential_keypoint <= min_extrema:
                keypoints.append((x, y))


    print(len(keypoints))
