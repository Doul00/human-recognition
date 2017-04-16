import cv2
import preprocessing
import hog

def main():
    img = cv2.imread('images/TIRF-3.jpg')

    img = preprocessing.grayscale(img)
    img = preprocessing.bilinear_resize(img, 64, 128)
    gradient_array = preprocessing.compute_gradient(img)
    magnitude_array = preprocessing.get_magnitude(img, gradient_array)
    direction_array = preprocessing.get_direction(img, gradient_array)
    img_histogram = hog.compute_all_histograms(magnitude_array, direction_array, 8)
    img_blocks = hog.create_blocks(img_histogram, 2, 8, img.shape[0], img.shape[1])
    hog_feature_vector = hog.concatenate_blocks(img_blocks)
    print(hog_feature_vector)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
