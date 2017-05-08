import math

import numpy as np


# ###########################################################################
# Filters
# ###########################################################################


def histogram_equalisation(img):
    """Color normalization to remove shadows and light effects."""
    rows = img.shape[0]
    columns = img.shape[1]
    for i in range(0, rows):
        for j in range(0, columns):
            blue = img.item(i, j, 0)
            green = img.item(i, j, 1)
            red = img.item(i, j, 2)

            total = float(blue + red + green)
            new_red = int(red / total * 255)
            new_green = int(green / total * 255)
            new_blue = int(blue / total * 255)
            img[i, j] = [new_blue, new_green, new_red]

    return img


def grayscale(img):
    """Basic grayscale filter."""
    rows = img.shape[0]
    columns = img.shape[1]
    for i in range(0, rows):
        for j in range(0, columns):
            blue = img.item(i, j, 0)
            green = img.item(i, j, 1)
            red = img.item(i, j, 2)

            gray = 0.299 * red + 0.587 * green + 0.114 * blue
            img[i, j] = [gray, gray, gray]
    return img


# ###########################################################################
# Resizing
# ###########################################################################


def interpolation(low_x, high_x, x, low_value, high_value):
    distance = high_x - low_x
    return (high_x - x)/distance * low_value + (x - low_x)/distance * high_value


def bilinear_resize(img, new_width, new_height):
    """Resize the image to 'new_width' and 'new_height' throught bilinear
    interpolation."""
    rows = img.shape[0]
    columns = img.shape[1]
    x_ratio = float(rows - 1) / new_height
    y_ratio = float(columns - 1) / new_width
    result = np.zeros((new_height, new_width, 3), np.uint8)
    for i in range(0, new_height):
        for j in range(0, new_width):
            x = x_ratio * i
            y = y_ratio * j

            floored_x = int(math.floor(x))
            ceiled_x = min(int(math.ceil(x + 1)), rows - 1) # Prevents out of bounds
            floored_y = int(math.floor(y))
            ceiled_y = min(int(math.ceil(y + 1)), columns - 1)
            r1 = interpolation(floored_y, ceiled_y, y,
                               img.item(floored_x, floored_y, 0),
                               img.item(floored_x, ceiled_y, 0))
            r2 = interpolation(floored_y, ceiled_y, y,
                               img.item(ceiled_x, floored_y, 0),
                               img.item(ceiled_x, ceiled_y, 0))
            new_value = int(interpolation(floored_x, ceiled_x, x, r1, r2))

            result[i][j] = [new_value, new_value, new_value]

    return result


# ###########################################################################
# Gradient Computation
# ###########################################################################


def is_in_bounds(i, j, img):
    """Boolean function returning True if the coordinates and inside the
    image."""
    rows = img.shape[0]
    columns = img.shape[1]
    return i >= 0 and i < rows and j >= 0 and j < columns


def compute_vertical_grad(i, j, img):
    """Vertical gradient with a mask [-1, 0, 1]'."""
    result = 0
    if is_in_bounds(i + 1, j, img):
        result += img.item(i + 1, j, 0) # Channel 0 because of grayscale.
    if is_in_bounds(i - 1, j, img):
        result -= img.item(i - 1, j, 0)
    return result


def compute_horizontal_grad(i, j, img):
    """Horizontal gradient with a mask [-1, 0, 1]."""
    result = 0
    if is_in_bounds(i, j + 1, img):
        result += img.item(i, j + 1, 0) # Channel 0 because of grayscale.
    if is_in_bounds(i, j - 1, img):
        result -= img.item(i, j - 1, 0)
    return result


def compute_gradient(img):
    """Create a 2d array of gradients. Each cell is made of the horizontal
    gradient and the vertical gradient."""
    rows = img.shape[0]
    columns = img.shape[1]
    gradient_array = [[ [0, 0] for j in range(columns)] for i in range(rows)]
    for i in range(0, rows):
        for j in range(0, columns):
            horizontal_grad = compute_horizontal_grad(i, j, img)
            vertical_grad = compute_vertical_grad(i, j, img)
            gradient_array[i][j] = [horizontal_grad, vertical_grad]
    return gradient_array


def get_magnitude(img, gradient_array):
    """Create a 2d array of the magnitudes. It is made of the square root of
    the addition of the power of two of the horizontal and vertical gradient
    of a same point. It represents the 'intensity' of both gradient."""
    rows = img.shape[0]
    columns = img.shape[1]
    magnitude_array = np.zeros((rows, columns), np.uint16)
    for i in range(0, rows):
        for j in range(0, columns):
            horizontal_grad = gradient_array[i][j][0]**2
            vertical_grad = gradient_array[i][j][1]**2
            magnitude_array[i][j] = math.sqrt(horizontal_grad + vertical_grad)
    return magnitude_array


def get_direction(img, gradient_array):
    """Create a 2d array of the directions. It is made of the arc tangent
    of the horizontal and vertical gradient of a same point. It represents
    the direction of both gradient. See 'contrast-normalization'."""
    rows = img.shape[0]
    columns = img.shape[1]
    angle_array = np.zeros((rows, columns), np.uint16)
    for i in range(0, rows):
        for j in range(0, columns):
            horizontal_grad = gradient_array[i][j][0]
            vertical_grad = gradient_array[i][j][1]
            computed_angle = math.degrees(math.atan2(vertical_grad,
                                                     horizontal_grad))
            if computed_angle < 0:
                computed_angle += 180
            angle_array[i][j] = computed_angle
    return angle_array

