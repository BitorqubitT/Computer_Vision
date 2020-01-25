# TODO: should check for input size of treshold,
# TODO: First: add numpy calc stuff, change names and clean up code.
# TODO: Third: test file that runs it x times on different size images check O() and times and plot in graph.
# TODO: Fourth: Use cpython to optimise if possible?
# TODO: FIFTH: Is it faster to check for threshold later on?
# TODO: SIXTH: Take the difference between two matrices.

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_path = "C:/Users/thier/Documents/Projects/Computer Vision/Lenna.png"

# Get image and set to numpy array in RGB
def getImage(image_path):
    image = cv2.imread(image_path)
    # Convert from bgr to rgb
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return img

# Convert image to Grayscale
# Y = 0.299 R0 + 0.587 G + 0.114 B
def rgbtoGray(matrix):
    new = np.ndarray(shape=(matrix.shape[0], matrix.shape[1]), dtype=float, order='F')

    for i in range(0, matrix.shape[0]):
        luminant = 0
        for j in range(0, matrix.shape[1]):
            luminant = matrix[i,j,0] * 0.299 + luminant + matrix[i,j,1] * 0.587 + luminant + matrix[i,j,2] * 0.114
            new[i,j] = (luminant/255)
            luminant = 0
    return new

# Sobel Filter
def sobel_x(matrix, threshold):
    Kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    result = np.zeros((matrix.shape[0]-2, matrix.shape[1]-2))
    for y in range(1, matrix.shape[0]-1):
        for x in range(1, matrix.shape[1]-1):
            result[y-1, x-1] = np.sum(np.multiply(matrix[y-1:y+2, x-1:x+2], Kernel))
            if result[y-1, x-1] < threshold:
                result[y-1, x-1] = 0
    return result

def sobel_y(matrix, threshold):
    Kernel = np.array([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]])
    result = np.zeros((matrix.shape[0]-2, matrix.shape[1]-2))
    for y in range(1, matrix.shape[0]-1):
        for x in range(1, matrix.shape[1]-1):
            result[y-1, x-1] = np.sum(np.multiply(matrix[y-1:y+2, x-1:x+2], Kernel))
            if result[y-1, x-1] < threshold:
                result[y-1, x-1] = 0
    return result

def combined(matrix, threshold):
    sobelx = sobel_x(matrix, threshold)
    sobely = sobel_y(matrix, threshold)
    result = np.sqrt(np.add(np.square(sobelx), np.square(sobely)))
    return result

# Create image from array
def createImage(matrix):
    # Does PIL rescale/normalize the values?
    # Maybe matplotlib better?
    # Might want to use a different mode.
    plt.imshow(matrix, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    return plt.show()

def main():
    xxxxx = rgbtoGray(getImage(image_path))
    loll = sobel_x(xxxxx, 0.2)
    lolll = sobel_y(xxxxx, 0.2)
    yo = combined(xxxxx, 0.2)
    yoo = combined(xxxxx, 0.7)

    createImage(yoo)
    createImage(yo)

if __name__ == "__main__":
    main()
