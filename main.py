import cv2
import preprocessing

def main():
    img = cv2.imread('images/TIRF-3.jpg')

    img = preprocessing.grayscale(img)
    img = preprocessing.bilinear_resize(img, 230, 300)
    gradient_array = preprocessing.compute_gradient(img)
    magnitude_array = preprocessing.get_magnitude(img, gradient_array)
    direction_array = preprocessing.get_direction(img, gradient_array)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
