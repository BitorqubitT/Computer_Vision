# TODO: Add y-sobel filter, combine them, get rid of ugly code, Add function for other filters.
# Sobel filter in python.
# Load image as matrix
# Convert to gray scale
# Perform sobel-operator
import cv2
from PIL import Image
import numpy as np

image_path = "C:/Users/thier/Documents/Projects/Computer Vision/Lenna.png"

# Get image and set to numpy array in RGB
def getImage(image_path):
    image = cv2.imread(image_path)
    # Convert from bgr to rgb
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return(img)

# Convert image to Grayscale
# Y = 0.299 R0 + 0.587 G + 0.114 B
def rgbtoGray(matrix):
    new = np.ndarray(shape=(matrix.shape[0], matrix.shape[1], 1), dtype=int, order='F')

    for i in range(0, matrix.shape[0]):
        luminant = 0
        for j in range(0, matrix.shape[1]):
            luminant = matrix[i,j,0] * 0.299 + luminant + matrix[i,j,1] * 0.587 + luminant + matrix[i,j,2] * 0.114
            new[i,j] = luminant
            luminant = 0
    return new

# Sobel Filter
def sobel(matrix):
    # Define sobel kernel
    Gx = np.array([[-1, 0, 1], [-1, 0, 1,], [-1, 0, 1]])
    Gy = np.array([[-1, -1, -1], [0, 0, 0,], [1, 1, 1]])

    new = np.ndarray(shape=(matrix.shape[0], matrix.shape[1], 1), dtype=int, order='F')
    # BLEHHHHHHHHHHH very ugly
    for i in range(0, matrix.shape[0] - 1):
        for j in range(0, matrix.shape[1] - 1):
            value1 = matrix[i-1, j-1] * 1
            value2 = matrix[i-1, j] * 0
            value3 = matrix[i-1, j+1] * -1
            value4 = matrix[i, j-1] * 1
            value5 = matrix[i, j] * 0
            value6 = matrix[i, j+1] * -1
            value7 = matrix[i+1, j-1] * 1
            value8 = matrix[i+1, j] * 0
            value9 = matrix[i+1, j+1] * -1
            value = value1 + value2 + value3 + value4 + value5 + value6 + value7 + value8 + value9
            new[i, j] = value
    return new

# Create image from array
def createImage(image):
    # Does PIL rescale/normalize the values?
    # Maybe matplotlib better?
    # Might want to use a different mode.
    image = Image.fromarray(image, mode = 'CMYK')
    image.show()

def main():
    x = rgbtoGray(getImage(image_path))
    done = sobel(x)
    createImage(done)

if __name__ == "__main__":
    main()
