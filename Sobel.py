# Sobel filter in python.
# Load image as matrix
# Convert to gray scale
# Perform sobel-operator
# Output

# Image uses PIL/ PILLOW
import cv2
from PIL import Image
import numpy as np

image_path = "C:/Users/thier/Documents/Projects/Computer Vision/Lenna.png"

# Get image and set to numpy array in RGB
def getImage(image_path):
    image = cv2.imread(image_path)
    # Convert from bgr to rgb
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(img, "lalalaal")
    return(img)



# Convert image to Grayscale
# Y = 0.299 R0 + 0.587 G + 0.114 B

def rgbtoGray(matrix):
    new = np.ndarray(shape=(matrix.shape[0], matrix.shape[1], 1), dtype=int, order='F')

    for i in range(0, matrix.shape[0]):
        luminant = 0
        for j in range(0, matrix.shape[1]):
            for k in range(0, matrix.shape[2]):
                if k is 0:
                    luminant = luminant + matrix[i,j,k] * 0.299
                elif k is 1:
                    luminant = luminant + matrix[i,j,k] * 0.587
                elif k is 2:
                    luminant = luminant + matrix[i,j,k] * 0.114
                    new[i,j] = luminant
                    luminant = 0
    print(new)
    return new

# Sobel Filter
def sobel(image):
    # Define sobel kernel
    Gx = np.array([[-1, 0, 1], [-1, 0, 1,], [-1, 0, 1]])
    Gy = np.array([[-1, -1, -1], [0, 0, 0,], [1, 1, 1]])

# Create image from array
def createImage(image):
    # Does PIL rescale/normalize the values?
    # Maybe matplotlib better?
    # Might want to use a different mode.
    image = Image.fromarray(image, mode = 'CMYK')

    image.show()

x = rgbtoGray(getImage(image_path))

createImage(x)
