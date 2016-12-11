#!/usr/bin/env python
"""
Computer Vision Project
Title: ??

Authors: Abhishek N. Kulkarni, Shravan Murlidaran, John Pryor
"""

# Imports
import cv2
import math
import numpy as np

# Templates for classification
PATH_TEMPLATE_DINO1 = 'dino1.txt'
PATH_TEMPLATE_DINO2 = 'dino2.txt'
PATH_TEMPLATE_DINO3 = 'dino3.txt'
PATH_TEMPLATE_DINO4 = 'dino4.txt'

PATH_TEXTURE_DINO1 = 'img/texture_dino1.jpg'
PATH_TEXTURE_DINO2 = 'img/texture_dino2.jpg'
PATH_TEXTURE_DINO3 = 'img/texture_dino3.jpg'
PATH_TEXTURE_DINO4 = 'img/texture_dino4.jpg'

dino1_template = None
dino2_template = None
dino3_template = None
dino4_template = None
templates = [dino1_template, dino2_template, dino3_template, dino4_template]

dino1_texture = None
dino2_texture = None
dino3_texture = None
dino4_texture = None
textures = dict()

# Color ranges
COLOR_HSV_RANGES = {'red': {'low': [], 'high': [], 'rgb': (0, 0, 255)},
                    'blue': {'low': [58, 33, 0], 'high': [163, 173, 255], 'rgb': (255, 0, 0)},
                    'yellow': {'low': [35, 100, 40], 'high': [55, 255, 255], 'rgb': (0, 255, 255)},
                    'green': {'low': [25, 40, 0], 'high': [142, 134, 253]}, 'rgb': (0, 255, 0)}
RED = 'red'
BLUE = 'blue'
GREEN = 'green'
YELLOW = 'yellow'

GAUSSIAN = 'gaussian filter'
MEDIAN = 'median filter'


def _segment_color(img, color, filt=None):
    """
    Performs color-based segmentation.

    :param img: RGB image
    :param color: Color from COLOR_COLOR_HSV_RANGES.
    :return: binary image
    """
    # Apply filter
    if filt == GAUSSIAN:
        filt_img = cv2.GaussianBlur(img, (11, 11), 5)
    elif filt == MEDIAN:
        filt_img = cv2.medianBlur(img, 5)
    else:
        filt_img = img

    # Convert to HSV
    hsv_img = cv2.cvtColor(filt_img, cv2.COLOR_BGR2HSV_FULL)

    # Extract color
    if color == RED:
        red_mask1 = cv2.inRange(hsv_img, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv_img, np.array([160, 100, 100]), np.array([179, 255, 255]))
        bin_img = cv2.bitwise_or(red_mask1, red_mask2)
    else:
        bin_img = cv2.inRange(hsv_img, np.array(COLOR_HSV_RANGES[color]['low']), np.array(COLOR_HSV_RANGES[color]['high']))

    # Perform Opening to remove small specks, closing for removing holes
    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    #bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    #bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    return bin_img


def preprocess_homography(img):
    """
    Preprocesses image to a binary images to detect color-coded end points.

    :param img: Raw RGB image
    :return: 4 corner points as 2-tuples in image coordinate [bottom-left, bottom-right, top-right, top-left]
    """
    # Detect required colors for corner points
    red = _segment_color(img, RED, GAUSSIAN)
    yellow = _segment_color(img, YELLOW, GAUSSIAN)
    blue = _segment_color(img, BLUE, GAUSSIAN)

    # Detect interest points: how?
    contours_red, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours
    contours_red = sorted(contours_red, key=cv2.contourArea, reverse=True)
    contours_yellow = sorted(contours_yellow, key=cv2.contourArea, reverse=True)
    contours_blue = sorted(contours_blue, key=cv2.contourArea, reverse=True)

    # Loop over contours to find
    for c in contours_red:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.07 * peri, True)
        if 5 > len(approx) > 3:
            print len(approx),"red"
            
            cv2.drawContours(img, [approx], -1, COLOR_HSV_RANGES[RED]['rgb'], 3)

    for c in contours_blue:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.07 * peri, True)
        if 5 > len(approx) > 3:
            print len(approx),"blue"
            cv2.drawContours(img, [approx], -1, COLOR_HSV_RANGES[BLUE]['rgb'], 3)

    for c in contours_yellow:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.07 * peri, True)
        
        if 5 > len(approx) > 3:
            print len(approx),"yellow"
            cv2.drawContours(img, [approx], -1, COLOR_HSV_RANGES[YELLOW]['rgb'], 3)

    return img


def main_test():
    # Setup Video Processing
    video = cv2.VideoCapture(0)

    while True:
        # Read Image
        _, img = video.read()
        r = _segment_color(img, RED, MEDIAN)
        y = _segment_color(img, YELLOW, MEDIAN)
        b = _segment_color(img, BLUE, MEDIAN)

        img = preprocess_homography(img)

        cv2.imshow('img', img)
        cv2.imshow('Red', r)
        cv2.imshow('Yellow', y)
        cv2.imshow('Blue', b)
        cv2.waitKey(3)
        if cv2.waitKey(3)== ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main_test()

