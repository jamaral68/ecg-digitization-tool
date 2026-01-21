import cv2 as cv
import numpy as np
import pandas as pd
from scipy import ndimage
from matplotlib import pyplot as plt
from PIL import Image
import skimage as ski

filename = 'bucket/img20250221_12515211'
image_name = filename + '.png'

image = cv.imread(image_name)

img_hsv=cv.cvtColor(image, cv.COLOR_BGR2HSV)

#Filter color to remove the grid
    
lower=(60,35,140) # black color
upper=(179,255,255) # dark gray
mask = cv.inRange(img_hsv, lower, upper)
result = img_hsv.copy()
result[mask!=255] = (255, 255, 255) # if it is not very dark set it to white

    #Convert to gray scale
image = cv.cvtColor(result, cv.COLOR_HSV2BGR )

plt.imshow(image, cmap="gray")
plt.show()
   
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

plt.imshow(image_gray, cmap="gray")
plt.show()