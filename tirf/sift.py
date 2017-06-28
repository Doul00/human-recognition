import math
import operator

import numpy as np
from scipy import misc
from scipy.ndimage import filters


def get_sift_descriptors(img):
    """
    Returns an array made of the concatenation of all descriptors. Each
    descriptor has a length of 128.
    """
    descriptors = []

    for descriptor in create_sift_descriptors(img).values():
        descriptors += descriptor

    return descriptors


def visualize_sift_descriptors(img, *, n=100):
    """
    Visualize the image with the sift descriptors.
    """
    from PIL import Image
    from PIL.ImageDraw import Draw
    image = Image.fromarray(img)
    draw = Draw(image)

    descriptors_magnitude = {
            coords: np.sum(descriptor)
            for coords, descriptor in create_sift_descriptors(img).items()
    }
    sorted_descriptors = sorted(descriptors_magnitude.items(),
                                key=operator.itemgetter(1))

    for ((x, y), _) in sorted_descriptors[-n:]:
        draw.ellipse([(x-2, y-2), (x+2, y+2)], fill='red')
    image.show()


def create_sift_descriptors(img):
    """
    See: http://docs.opencv.org/trunk/da/df5/tutorial_py_sift_intro.html
    """
    keypoints = difference_of_gaussian(img)
    keypoints &= filter_low_contrast(img, keypoints)
    #keypoints &= filter_harris_points(img, keypoints)

    return create_descriptors(img, keypoints)


def difference_of_gaussian(img):
    octaves  = compute_octaves(img)
    octaves = [difference_of_blur(octave) for octave in octaves]

    keypoints = set()
    for scale_level, octave in enumerate(octaves):
        for i in range(len(octave)-2):
            keypoints |= {(x * 2 ** scale_level, y * 2 ** scale_level)
                          for (x, y) in find_keypoints_extrema(octave[i:i+3])}

    return keypoints


def difference_of_blur(octave):
    """
    Returns an absolute difference between images two by two. Each image
    belongs to the same octave (i.e. same scale) but has a different blur
    level.
    """
    # We need to do a conversion to uint8 (unsigned) to int16 (signed)
    # when doing the difference, or the results are fucked.
    octave = [np.int16(img) for img in octave]
    diffs  = [np.abs(octave[i] - octave[i+1]) for i in range(len(octave)-1)]
    octave = [np.uint8(img) for img in diffs]

    return octave


def compute_octaves(img, *, octave_nb=4):
    """
    Computes @octave_nb octaves, the first image has the original size. The
    followings are scaled down by 50% at each step.
    """
    octaves = [add_blur_levels(img)]

    for _ in range(octave_nb-1):
        img = scale_down(img)
        octaves.append(add_blur_levels(img))

    return octaves


def add_blur_levels(img, *, sigma=1.6, octave_size=5):
    """
    Computes @octave_size blurred images of a same octave with a @sigma
    starting from 1.6 and growing each time by a factor a sqrt(2).
    """
    blur_levels = []

    for i in range(octave_size):
        blur_levels.append(filters.gaussian_filter(img, sigma))
        sigma *= math.sqrt(2)

    return blur_levels


def scale_down(img):
    """
    Scales down the @img to half its original size.
    """
    return misc.imresize(img, .5)


def find_keypoints_extrema(octave):
    """
    Local extrema (max or min) are selected as potential keypoints between
    a "cube" made of 3 layers of blur levels of a same octave.
    """
    mid = len(octave) // 2
    keypoints = set()

    rows    = octave[0].shape[0]
    columns = octave[0].shape[1]
    for y in range(1, rows-1): # We are skipping the border pixels as they
        for x in range(1, columns-1): # probably won't be keypoints.

            cube_center = octave[mid].item(y, x, 0)
            flag_is_max = True
            flag_is_min = True

            for yy in range(-1, 2):
                for xx in range(-1, 2):
                    coord_y = y + yy
                    coord_x = x + xx

                    for i in range(len(octave)):
                        if i == mid and\
                           coord_x == x and coord_y == y:
                            continue

                        if octave[i].item(coord_y, coord_x, 0) >= cube_center:
                            flag_is_max = False
                        if octave[i].item(coord_y, coord_x, 0) <= cube_center:
                            flag_is_min = False

            if flag_is_max or flag_is_min:
                keypoints.add((x, y))

    return keypoints


def compute_harris_score(img, *, sigma=1.6):
    """
    Computes the harris score (from the harris corner detection algorithm)
    for each pixel.
    See for all the related notation:
        http://aishack.in/tutorials/harris-corner-detector
    """
    # First we are computing the derivating for each axis (x, and y) found
    # with the Taylor series.
    i_x = filters.gaussian_filter1d(img, sigma=sigma, order=1, axis=0)
    i_y = filters.gaussian_filter1d(img, sigma=sigma, order=1, axis=1)

    i_xx = filters.gaussian_filter(i_x * i_x, sigma)
    i_yy = filters.gaussian_filter(i_y * i_y, sigma)
    i_xy = filters.gaussian_filter(i_x * i_y, sigma)

    # We are now computing the score R for a window with the eigenvalues of
    # the matrix:
    #   M = [ Ix^2  IxIy ]
    #       [ IxIy  Iy^2 ]
    # With R = det M - trace(M)^2

    det_m = i_xx * i_yy - i_xy ** 2
    tr_m  = i_xx + i_yy

    return det_m - tr_m ** 2


def filter_harris_points(img, keypoints, *, threshold=10):
    """
    Given a harris corner score for each pixel position, we are filtering
    all pixel' scores that are above the @threshold.
    """
    harris_scores = compute_harris_score(img)
    filtered_coords = set()

    for (x, y) in keypoints:
        if harris_scores.item(y, x, 0) < threshold:
            filtered_coords.add((x, y))

    return filtered_coords


def filter_low_contrast(img, keypoints, *, threshold=0.03):
    """
    Filter all keypoints that have a low contrast.
    """
    filtered_coords = set()

    for (x, y) in keypoints:
        window = img[y-1:y+2, x-1:x+2, 0]

        std = window.std()
        if std == 0:
            continue
        mean = window.mean()

        if abs((img.item(y, x, 0) - mean) / std) > threshold:
            filtered_coords.add((x, y))

    return filtered_coords


def create_descriptors(img, keypoints):
    """
    Creates a descriptor for each keypoint. It is returned under the form of
    a dictionnary of keypoint coordinates as key, and its associated
    descriptor as value.
    """
    return {keypoint: compute_descriptor(img, keypoint)
            for keypoint in keypoints}


def compute_descriptor(img, keypoint):
    """
    Computes a descriptor with the given keypoint.
    The descriptor is a numerical vector of lenght 128. It is the
    concatenation of 16 subblocks' histograms from the surronding 16x16
    neighbours pixels. Each subblock has a size of 4x4.
    """
    x, y = keypoint
    histograms = []

    subblock_start = [
            (-8, -8), (-4, -8), (0, -8), (4, -8),
            (-8, -4), (-4, -4), (0, -4), (4, -4),
            (-8,  0), (-4,  0), (0,  0), (4,  0),
            (-8,  4), (-4,  4), (0,  4), (8,  4)
    ]

    for (start_x, start_y) in subblock_start:
        histograms += create_histogram(img, x + start_x, y + start_y)

    return histograms


def create_histogram(img, x, y):
    """
    Creates an histogram for the given subblock.
    @x and @y are the upper left coordinates of the subblock. The histogram
    is divided in 8 bins, ranging from -pi/2 to pi/2.
    """
    nb_bins = 8
    bins = [0 for _ in range(nb_bins)]

    pi_2 = math.pi / 2
    pi_8 = math.pi / 8

    rows    = img.shape[0]
    columns = img.shape[1]

    for yy in range(y, y+5):
        for xx in range(x, x+5):
            if (xx < -1 or xx >= columns-1) or (yy < -1 or yy >= rows-1):
                break

            magnitude = compute_magnitude(img, (xx, yy))
            direction = compute_direction(img, (xx, yy))

            for i in range(len(bins)):
                if direction < (-pi_2 + (i+1) * pi_8):
                    bins[i] += magnitude
                    break

    return bins


def compute_direction(img, coords):
    """
    Computes the direction for surrounding pixels.
    """
    x, y = coords
    return math.atan2((img.item(y+1, x, 0) - img.item(y-1, x, 0)),
                      (img.item(y, x+1, 0) - img.item(y, x-1, 0)))


def compute_magnitude(img, coords):
    """
    Computes the magnitude for surrounding pixels.
    """
    x, y = coords
    return math.sqrt((img.item(y, x-1, 0) - img.item(y, x+1, 0)) ** 2 +\
                     (img.item(y-1, x, 0) - img.item(y+1, x, 0)) ** 2)
