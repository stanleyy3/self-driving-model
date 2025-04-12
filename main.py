import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

from moviepy.editor import VideoFileClip
from IPython.display import HTML

import os
os.listdir("test_images/")

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
#    plt.figure()     #For Debug
#    plt.imshow(masked_image)      #For Debug
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """ 
    posSlopePoints = []
    negSlopePoints = []
    bestPosLineFitSlopeInt =[]
    bestNegLineFitSlopeInt= []
            
    for line in lines:
        for x1,y1,x2,y2 in line:
        # Find the Slope ((y2-y1)/(x2-x1))
        # If slope > 0, it should be the right lane line & if slope < 0, it should be left lane
            slope = ((y2-y1)/(x2-x1))
        # Filter others by picking a range for slope of lane lines
#            print("Slope:", Slope)
        # Get all lines and store the positive and negative slope line points separately
            if 0.85 > slope > 0.45: 
                if not math.isnan(x1) or math.isnan(y1) or math.isnan(x2) or math.isnan(y2):
                    posSlopePoints.append([x1, y1])
                    posSlopePoints.append([x2, y2])
            # Filter others by picking a range for slope of lane lines
            elif -0.85 < slope < -0.45: 
                if not math.isnan(x1) or math.isnan(y1) or math.isnan(x2) or math.isnan(y2):
                    negSlopePoints.append([x1, y1])
                    negSlopePoints.append([x2, y2])                

#    print("Positive slope line points: ", posSlopePoints)
#    print("Negative slope line points: ", negSlopePoints)
        
#    posSlopeXs = []
#    posSlopeYs = []
#    negSlopeXs = []
#    negSlopeYs = []
    
    posSlopeXs = [pair[0] for pair in posSlopePoints]
    posSlopeYs = [pair[1] for pair in posSlopePoints]
    negSlopeXs = [pair[0] for pair in negSlopePoints]
    negSlopeYs = [pair[1] for pair in negSlopePoints]

# Get the best line fit through the available points and store the slope & intercept of this line for both left & right lanes   
    bestPosLineFitSlopeInt=np.polyfit(posSlopeXs, posSlopeYs, 1)
    bestNegLineFitSlopeInt=np.polyfit(negSlopeXs, negSlopeYs, 1)

# Once we have line which is the best available fit through all the lines, extend this line to the ROI mask edges
# Extended Left lane line bottom co-ordinates
    leftby = imgheight  # Y coordinate from ROI mask
    leftbx = (leftby-bestNegLineFitSlopeInt[1])/bestNegLineFitSlopeInt[0]  # X coordinate = (y-c)/m
    
# Extended Left lane line top co-ordinates
    leftty = 0.62*imgheight  # Y coordinate from ROI mask
    lefttx = (leftty-bestNegLineFitSlopeInt[1])/bestNegLineFitSlopeInt[0]  # X coordinate = (y-c)/m

# Extended right lane line bottom co-ordinates
    rightby = imgheight  # Y coordinate from ROI mask
    rightbx = (rightby-bestPosLineFitSlopeInt[1])/bestPosLineFitSlopeInt[0]  # X coordinate = (y-c)/m
    
# Extended right lane line top co-ordinates
    rightty = 0.62*imgheight  # Y coordinate from ROI mask
    righttx = (rightty-bestPosLineFitSlopeInt[1])/bestPosLineFitSlopeInt[0]  # X coordinate = (y-c)/m

    cv2.line(img, (int(leftbx),int(leftby)), (int(lefttx), int(leftty)), color, thickness)
#    plt.figure()
#    plt.imshow(img)  #For debug
    cv2.line(img, (int(rightbx), int(rightby)), (int(righttx), int(rightty)), color, thickness)  
#    plt.imshow(img)  #For debug


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

'''image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)
plt.show()'''

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

# Read in and grayscale the image
# image = mpimg.imread('test_images\whiteCarLaneSwitch.jpg')
def process_static_image(image):

#    plt.figure(figsize=(52.5,100))
#    plt.subplots_adjust(wspace=0.1, hspace=0.1)
#    plt.subplot(231)
#    plt.imshow(image)  #For debug

#    hsvimage = to_hsv(image)
    gray = grayscale(image)
#    plt.imshow(gray, cmap="gray")
    
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
#    blur_hsv = gaussian_blur(hsvimage, kernel_size)
    blur_gray = gaussian_blur(gray, kernel_size)
    
#    plt.subplot(2, 3, 2)
#    plt.imshow(blur_gray, cmap="gray")
    
    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
#    edges = canny(blur_hsv, low_threshold, high_threshold)
    edges = canny(blur_gray, low_threshold, high_threshold)

#    plt.subplot(2, 3, 3)
#    plt.imshow(edges, cmap="gray")

#    Tried the dilated approach but found out it wasn't good
#    dilated_edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_DILATE, (5, 5)))
#    plt.imshow(dilated_edges, cmap="gray")
    
    # Defining a four sided polygon to create a mask
    global imgheight
    global imgwidth
    imgheight = image.shape[0]
    imgwidth = image.shape[1]
    vertices = np.array([[(0.05*imgwidth,imgheight),(0.48*imgwidth, 0.62*imgheight), (0.55*imgwidth, 0.62*imgheight), (0.95*imgwidth,imgheight)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
#    masked_edges = region_of_interest(dilated_edges, vertices)  #Use with dilate approach only

#    plt.subplot(2, 3, 4)
#    plt.imshow(masked_edges, cmap="gray")  #For debug

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180  # angular resolution in radians of the Hough grid
    threshold = 40     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 10   # minimum number of pixels making up a line
    max_line_gap = 50  # maximum gap in pixels between connectable line segments

    hough_lines_img = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
#    plt.subplot(2, 3, 5)                 #For debug
#    plt.imshow(hough_lines_img, cmap="gray")  #For debug

    line_marked_img=weighted_img(hough_lines_img, image, α=0.8, β=1., λ=0.)#, 3, 6)                 
#    plt.subplot(2,3,6)           #For debug
#    plt.imshow(line_marked_img)  #For debug

    return line_marked_img

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = process_static_image(image)

    return result

inputDir = "test_images"
outputDir = inputDir + "_out"
if not os.path.exists(outputDir):
    os.makedirs(outputDir)
imgTitles = os.listdir(inputDir + "/")
for imgTitle in imgTitles:
    image = mpimg.imread(inputDir + "/" + imgTitle)
#    plt.figure(figsize=(52.5, 100))
#    plt.subplot(121)
#    plt.imshow(image)  #For debug
    outputImg = process_static_image(image)
#    plt.subplot(122)
#    plt.imshow(outputImg)  #For debug
    mpimg.imsave(outputDir + "/" + imgTitle, outputImg)
    print("Processed " + outputDir + "/" + imgTitle)
