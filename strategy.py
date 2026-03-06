import cv2 as cv
from edt_utils import extract_image 


def color(image, lower, upper, thres_value):
    img_hsv=cv.cvtColor(image, cv.COLOR_BGR2HSV)
    #Filter color to remove the grid
    #lower=(0,0,0) # black colssor
    #upper=(179,255,220) # dark gray
    mask = cv.inRange(img_hsv, lower, upper)
    result = img_hsv.copy()
    result[mask!=255] = (255, 255, 255) # if it is not very dark set it to white
    #Convert to gray scale
    image = cv.cvtColor(result, cv.COLOR_HSV2BGR )
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # To binary image
    ret, th1 = cv.threshold(image_gray, thres_value, 255,cv.THRESH_BINARY)
    return ret, th1

def filter(image, kSize2d, kSize1d, thres_value):
    image_gray = extract_image(image, kSize2d, kSize1d)
    ret, th1 = cv.threshold(image_gray, thres_value, 255,cv.THRESH_BINARY) # transform to binary
    return ret, th1    

def none(image, thres_value):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, th1 = cv.threshold(image_gray, thres_value, 255,cv.THRESH_BINARY) # transform to binary
    return ret, th1