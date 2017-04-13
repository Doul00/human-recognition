import math

#
# Filters
#

# Color normalization to remove shadows and light effects
def histogram_equalisation(img):
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

# Simple grayscale filter
def grayscale(img):
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

#
# Gradient Computation
#

def is_in_bounds(i, j, img):
    rows = img.shape[0]
    columns = img.shape[1]
    if (i >= 0 and i < rows and j >= 0 and j < columns):
        return True
    else:
        return False

def compute_horizonal_grad(i, j, img):
    result = 0
    if is_in_bounds(i + 1, j, img):
        result += img.item(i + 1, j, 0) # Channel 0 because of grayscale
    if is_in_bounds(i - 1, j, img):
        result -= img.item(i - 1, j, 0)
    return result

def compute_vertical_grad(i, j, img):
    result = 0
    if is_in_bounds(i, j + 1, img):
        result += img.item(i, j + 1, 0) # Channel 0 because of grayscale
    if is_in_bounds(i, j - 1, img):
        result -= img.item(i, j - 1, 0)
    return result

def compute_gradient(img):
    rows = img.shape[0]
    columns = img.shape[1]
    gradient_array = [[ [0, 0] for j in range(columns)] for i in range(rows)]
    for i in range(0, rows):
        for j in range(0, columns):
            horizontal_grad = compute_horizonal_grad(i, j, img)
            vertical_grad = compute_vertical_grad(i, j, img)
            gradient_array[i][j] = [horizontal_grad, vertical_grad]
    return gradient_array

def get_magnitude(img, gradient_array):
    rows = img.shape[0]
    columns = img.shape[1]
    magnitude_array = [[ 0 for j in range(columns)] for i in range(rows)]
    for i in range(0, rows):
        for j in range(0, columns):
            x_val = gradient_array[i][j][0]**2
            y_val = gradient_array[i][j][1]**2
            magnitude_array[i][j] = math.sqrt(x_val + y_val)
    return magnitude_array

def get_direction(img, gradient_array):
    rows = img.shape[0]
    columns = img.shape[1]
    angle_array = [[ 0 for j in range(columns)] for i in range(rows)]
    for i in range(0, rows):
        for j in range(0, columns):
            x_val = gradient_array[i][j][0]
            y_val = gradient_array[i][j][1]
            computed_angle = math.degrees(math.atan(y_val / x_val))
            if computed_angle < 0:
                computed_angle += 180
            angle_array[i][j] = computed_angle
    return angle_array
