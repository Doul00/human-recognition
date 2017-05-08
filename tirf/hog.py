import math

import numpy as np

from .cell import Cell, Block


def normalize_vector(vect):
    """Basic normalization of a vector."""
    total = sum(elt ** 2 for elt in vect)
    norm = math.sqrt(total)
    if int(norm) == 0:
        norm = 1 # Workaround hack, to fix later
    vect = vect / norm
    return vect


def compute_cell_histogram(magnitude_array, direction_array, offset_x,
                           offset_y, cell_index, cell_size):
    """Computes the histogram of a cell_size*cell_size cell."""
    histogram = np.zeros((9), np.uint16)
    for i in range(offset_x, offset_x + cell_size):
        for j in range(offset_y, offset_y + cell_size):
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


def compute_all_histograms(magnitude_array, direction_array, cell_size):
    """Computes histograms for all of the cells."""
    cell_index = 0
    offset_x, offset_y = 0, 0
    rows, cols = magnitude_array.shape[0], magnitude_array.shape[1]
    picture_cells = [None] * int((rows * cols / cell_size ** 2))

    while offset_x <= rows - cell_size and offset_y <= cols - cell_size:
        new_cell = compute_cell_histogram(magnitude_array, direction_array,
                                          offset_x, offset_y, cell_index,
                                          cell_size)

        picture_cells[cell_index] = new_cell
        cell_index += 1
        if offset_y == cols - cell_size and offset_x != rows - cell_size:
            offset_y = 0
            offset_x += cell_size
        else:
            offset_y += cell_size

    return picture_cells


def create_blocks(picture_cells, cells_per_block, cell_size, rows, cols):
    """Creates overlapping blocks with size = cells_per_block * cell_size."""
    block_number = 0
    block_array = []
    cells_per_line = int(cols / cell_size)
    cells_per_column = int(rows / cell_size)

    # Remove cell_per_block so the block won't go outside of bounds
    for i in range (cells_per_column - (cells_per_block - 1)):
        for j in range (cells_per_line - (cells_per_block - 1)):
            concatenated_vect = np.asarray([])
            rows_index = i * cells_per_line
            line_cells = np.asarray([x for x in range(rows_index + j,
                                    rows_index + cells_per_block + j)])
            block_matrix = np.asarray([]).astype(int)
            block_matrix = np.append(block_matrix, line_cells)
            down_cells_index = cells_per_line

            # Computes the values for the block matrix
            for k in range(cells_per_block - 1):
                block_matrix = np.append(block_matrix,
                                         np.copy(line_cells) + down_cells_index)
                down_cells_index += cells_per_line

            # Gets the histogram for each value in the matrix
            for index in block_matrix:
                concatenated_vect = np.append(concatenated_vect,
                                              picture_cells[index].histogram)

            concatenated_vect = normalize_vector(concatenated_vect)
            block_array += [Block(concatenated_vect)]
            block_number += 1
    return np.asarray(block_array)


def concatenate_blocks(block_array):
    """Create the final feature vector."""
    feature_vector = np.asarray([])
    for block in block_array:
        feature_vector = np.append(feature_vector, block.histogram)
    return feature_vector

