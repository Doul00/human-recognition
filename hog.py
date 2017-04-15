import numpy as np
import math
from cell import Cell

def compute_cell_histogram(magnitude_array, direction_array, offset_x,
                           offset_y, cell_index):

    histogram = np.zeros((9), np.uint16)
    for i in range(offset_x, offset_x + 8):
        for j in range(offset_y, offset_y + 8):
            grad = direction_array[i][j]
            magnitude = magnitude_array[i][j]

            lower_bound = math.floor(grad / 20) * 20
            upper_bound = math.ceil(grad / 20) * 20
            low_contrib = (20 - (grad - lower_bound)) / 20 * magnitude
            high_contrib = (20 - (upper_bound - grad)) / 20 * magnitude

            if upper_bound == 180:
                histogram[0] += high_contrib
            else:
                histogram[int(upper_bound / 20)] += high_contrib

            if grad != 180: # Prevents adding 180° angle twice
                histogram[int(lower_bound / 20)] += low_contrib

    return Cell(cell_index, histogram)

def compute_all_histograms(magnitude_array, direction_array):
    picture_cells = [Cell] * 128
    cell_index = 0
    offset_x = 0
    offset_y = 0
    rows = magnitude_array.shape[0]
    cols = magnitude_array.shape[1]

    while offset_x <= rows - 8 and offset_y <= cols - 8:
        new_cell = compute_cell_histogram(magnitude_array, direction_array,
                                          offset_x, offset_y, cell_index)

        picture_cells[cell_index] = new_cell
        cell_index += 1
        if offset_y == cols - 8 and offset_x != rows - 8:
            offset_y = 0
            offset_x += 8
        else:
            offset_y += 8

    return picture_cells
