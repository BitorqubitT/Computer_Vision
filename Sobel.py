# Sobel filter in python.
# Load image as matrix
# Perform sobel-operator
# Output

# Image uses PIL/ PILLOW
import cv2
from PIL import Image
import numpy as np

image_path = "C:/Users/thier/Documents/Projects/Computer Vision/Lenna.png"

image = cv2.imread(image_path)
# Convert from bgr to rgb
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Get image and set to numpy array in RGB
def getImage(image_path):
    image = cv2.imread(image_path)
    # Convert from bgr to rgb
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return(img[2,3])


# Sobel Filter
def sobel(image):
    # Define sobel kernel
    Gx = []
    Gy = []

    getimagey
    getimagex

    # For each px r,c
    # convolve Gx and Gy

    #  gradient = squart(Gx^2 + Gy^2)
    for









# Create image from array
def createImage():
    # Does PIL rescale/normalize the values?
    # Maybe matplotlib better?
    image = Image.fromarray(img, 'RGB')
    image.save('LOL.png')
    image.show()




print(getImage(image_path))
