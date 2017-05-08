import time
import math

from PIL import Image
from PIL.ImageDraw import Draw
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
    original_shape = img.shape
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


    if verbose:
        visualize_hog(hog_feature_vector, img)
    return hog_feature_vector


def visualize_hog(hog_features, img):
    dim_width  = img.shape[1]
    dim_height = img.shape[0]

    zoom = 3
    img = Image.fromarray(img)

    cell_size = 8
    bin_size  = 9
    rad_range = 180 / 9

    nb_width  = dim_width // cell_size
    nb_height = dim_height // cell_size

    gradients_strength = [[[.0 for _ in range(bin_size)]
                           for _ in range(nb_width)]
                          for _ in range(nb_height)]

    cell_update_counter = [[0 for _ in range(nb_width)]
                           for _ in range(nb_height)]

    hog_index = 0

    for block_w in range(nb_width - 1):
        for block_h in range(nb_height - 1):
            for cell in range(4):
                cell_w = block_w
                cell_h = block_h
                if cell == 1:
                    cell_h += 1
                elif cell == 2:
                    cell_w += 1
                elif cell == 3:
                    cell_w += 1
                    cell_h += 1

                for b in range(bin_size):
                    gradient_strength = hog_features[hog_index]
                    hog_index += 1
                    gradients_strength[cell_h][cell_w][b] += gradient_strength

                cell_update_counter[cell_h][cell_w] += 1


    for cell_w in range(nb_width):
        for cell_h in range(nb_height):
            nb_update = cell_update_counter[cell_h][cell_w]

            for b in range(bin_size):
                gradients_strength[cell_h][cell_w][b] /= nb_update


    draw = Draw(img)
    for cell_w in range(nb_width):
        for cell_h in range(nb_height):
            draw_x = cell_w * cell_size
            draw_y = cell_h * cell_size

            my_x = draw_x + cell_size / 2
            my_y = draw_y + cell_size / 2

            """
            draw.rectangle([(draw_x, draw_y),
                            (draw_x+cell_size, draw_y+cell_size)],
                           outline=128)
            """

            for b in range(bin_size):
                grad = gradients_strength[cell_h][cell_w][b]
                if grad == 0:
                    continue

                rad = b * rad_range + rad_range/2
                rad_x = math.cos(rad)
                rad_y = math.sin(rad)
                max_vec_len = cell_size/2
                scale = 2.5

                x0 = my_x - rad_x * grad * max_vec_len * scale
                y0 = my_y - rad_y * grad * max_vec_len * scale
                x1 = my_x + rad_x * grad * max_vec_len * scale
                y1 = my_y + rad_y * grad * max_vec_len * scale

                draw.line([(x0, y0), (x1, y1)], fill="red")

    img = img.resize((128, 256))
    img.show()
