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
                    'yellow': {'low': [35, 100, 0], 'high': [55, 255, 255], 'rgb': (0, 255, 255)},
                    #'yellow': {'low': [27, 59, 11], 'high': [45, 255, 255], 'rgb': (0, 255, 255)},
                    'green': {'low': [25, 40, 0], 'high': [142, 134, 253]}, 'rgb': (0, 255, 0)}
RED = 'red'
BLUE = 'blue'
GREEN = 'green'
YELLOW = 'yellow'

GAUSSIAN = 'gaussian filter'
MEDIAN = 'median filter'

# Define filter
SMA_WINDOW = 10
filt_state = {'p0': [(0, 0)] * SMA_WINDOW,
              'p1': [(0, 0)] * SMA_WINDOW,
              'p2': [(0, 0)] * SMA_WINDOW,
              'p3': [(0, 0)] * SMA_WINDOW}


def load_templates():
    """
    Loads templates of dinosaurs from predefined global file paths stored in
    PATH_DINO1, PATH_DINO2, PATH_DINO3, PATH_DINO4.

    :return: Boolean = True if load is successful. Else False.
    """
    global dino1_template, dino2_template, dino3_template, dino4_template

    try:
        # Read templates
        dino1 = np.loadtxt(PATH_TEMPLATE_DINO1, np.uint8)
        dino2 = np.loadtxt(PATH_TEMPLATE_DINO2, np.uint8)
        dino3 = np.loadtxt(PATH_TEMPLATE_DINO3, np.uint8)
        dino4 = np.loadtxt(PATH_TEMPLATE_DINO4, np.uint8)

        # Retrieve contours
        dino1_template, _ = cv2.findContours(dino1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dino2_template, _ = cv2.findContours(dino2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dino3_template, _ = cv2.findContours(dino3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dino4_template, _ = cv2.findContours(dino4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        templates = [dino1_template, dino2_template, dino3_template, dino4_template]
        return True

    except IOError:
        print 'Templates could not be loaded. Check file paths'

    return False


def load_textures():
    """
    Loads templates of dinosaurs from predefined global file paths stored in
    PATH_DINO1, PATH_DINO2, PATH_DINO3, PATH_DINO4.

    :return: Boolean = True if load is successful. Else False.
    """
    global dino1_texture, dino2_texture, dino3_texture, dino4_texture, textures

    try:
        # Read texture files
        dino1_texture = cv2.imread(PATH_TEXTURE_DINO1)
        dino2_texture = cv2.imread(PATH_TEXTURE_DINO2)
        dino3_texture = cv2.imread(PATH_TEXTURE_DINO3)
        dino4_texture = cv2.imread(PATH_TEXTURE_DINO4)

        textures = {0: dino1_texture, 1: dino2_texture, 2: dino3_texture, 3: dino4_texture}
        return True

    except IOError:
        print 'Textures could not be loaded. Check file paths'

    return False


def ccw(p1, p2, p3):
    """Tests whether the turn formed by A, B, and C is ccw"""
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) > (p2[1] - p1[1]) * (p3[0] - p1[0])


def sma(queue, new_val):
    N = len(queue)
    first = queue[0]
    queue.append(queue[-1] + float(new_val) / N - float(first) / N)
    return queue


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


def _filt_hgrf_points(points):
    """
    Applies SMA filter on points to remove high frequencies from the image.

    :param points: points dictionary in format from preprocess_homography function.
    :return:
    """
    global filt_state

    # get points
    #print ' points: ', points
    red = points['red']
    yellow = points['yellow']
    blue1 = points['blue1']
    blue2 = points['blue2']

    # Check validity
    if len(blue1) == 0 or len(blue2) == 0 or len(red) == 0 or len(yellow) == 0:
        return filt_state['p0'][0], filt_state['p1'][0], filt_state['p2'][0], filt_state['p3'][0]

    else:
        # Detect which blue corresponds to which point
        if ccw(blue1, blue2, red):
            p1 = blue1
            p3 = blue2
        else:
            p1 = blue2
            p3 = blue1

        # Assign other points
        p0 = red
        p2 = yellow
        p = [tuple(p0), tuple(p1), tuple(p2), tuple(p3)]

        # Update State
        for i in range(4):
            x = [k[0] for k in filt_state['p' + str(i)]]
            y = [k[1] for k in filt_state['p' + str(i)]]

            x = int(sma(x, p[i][0])[-1])
            y = int(sma(y, p[i][1])[-1])

            filt_state['p' + str(i)].append((x, y))

        return filt_state['p0'].pop(0), filt_state['p1'].pop(0), filt_state['p2'].pop(0), filt_state['p3'].pop(0)


def preprocess_homography(img):
    """
    Preprocesses image to a binary images to detect color-coded end points.

    :param img: Raw RGB image
    :return: 4 corner points as 2-tuples in image coordinate [bottom-left, bottom-right, top-right, top-left]
    """
    # Initialize variables
    hgrf_img = img.copy()
    hgrf_points = {'red': [], 'blue1': [], 'blue2': [], 'yellow': []}

    # Detect required colors for corner points
    red = _segment_color(hgrf_img, RED, GAUSSIAN)
    yellow = _segment_color(hgrf_img, YELLOW, GAUSSIAN)
    blue = _segment_color(hgrf_img, BLUE, GAUSSIAN)

    # Detect interest points: how?
    contours_red, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours
    contours_red = sorted(contours_red, key=cv2.contourArea, reverse=True)
    contours_yellow = sorted(contours_yellow, key=cv2.contourArea, reverse=True)
    contours_blue = sorted(contours_blue, key=cv2.contourArea, reverse=True)

    # RED
    try:
        contours_red = max([c for c in contours_red if (30 < cv2.contourArea(c) < 1500
                            and 3 <= len(cv2.approxPolyDP(c, 0.07 * cv2.arcLength(c, True), True)) <= 5)],
                           key=cv2.contourArea)
        cv2.drawContours(hgrf_img, [contours_red], -1, COLOR_HSV_RANGES[RED]['rgb'], 3)

        # Get Centroid
        M = cv2.moments(contours_red)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        hgrf_points['red'] = [cx, cy]

    except ValueError, e:
        print 'error!', e

    # YELLOW
    try:
        contours_yellow = max([c for c in contours_yellow if (30 < cv2.contourArea(c) < 1500
                            and 3 <= len(cv2.approxPolyDP(c, 0.07 * cv2.arcLength(c, True), True)) <= 5)],
                           key=cv2.contourArea)
        cv2.drawContours(hgrf_img, [contours_yellow], -1, COLOR_HSV_RANGES[YELLOW]['rgb'], 3)

        # Get Centroid
        M = cv2.moments(contours_yellow)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        hgrf_points['yellow'] = [cx, cy]
    except ValueError, e:
        print 'error!', e

    # BLUE
    try:
        contours_blue = sorted([c for c in contours_blue if (30 < cv2.contourArea(c) < 1500
                            and 3 <= len(cv2.approxPolyDP(c, 0.07 * cv2.arcLength(c, True), True)) <= 5)],
                           key=cv2.contourArea, reverse=True)
        cv2.drawContours(hgrf_img, [contours_blue[0]], -1, COLOR_HSV_RANGES[BLUE]['rgb'], 3)
        cv2.drawContours(hgrf_img, [contours_blue[1]], -1, COLOR_HSV_RANGES[BLUE]['rgb'], 3)

        cnt1 = contours_blue[0]
        M = cv2.moments(cnt1)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        hgrf_points['blue1'] = [cx, cy]

        cnt2 = contours_blue[1]
        M = cv2.moments(cnt2)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        hgrf_points['blue2'] = [cx, cy]

    except ValueError, e:
        print 'error!', e
    except IndexError, e:
        print 'error!', e

    return hgrf_img, hgrf_points


def applyTransform(img, points):
    """
    Applies perspective transformation.
    :param points: 4-list of 2-tuples.
    :return: transformed image
    """
    # Define new image dimensions
    height = 640
    width = int(height * 0.75)
    width, height = width // 10 * 8, height // 10 * 8

    # Define destination points
    dst_points = np.array([[height, 0], [height, width], [0, width], [0, 0]], dtype=np.float32)
    #print np.array(points), dst_points

    # Apply transformation
    perspective_matrix = cv2.getPerspectiveTransform(np.array(points, dtype=np.float32), dst_points)
    tr_img = cv2.warpPerspective(img, perspective_matrix, (height, width))

    return tr_img


def preprocess_dinos(img):
    """
    Preprocess image for detecting dinosaurs.

    :param img: RGB Image
    :return: 2-tuple of (binary image, contours)
    """
    # Color selection
    hl = 69; sl = 26; vl = 0; hu = 126; su = 255; vu = 255

    # Filter Image
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    blur_img = cv2.GaussianBlur(img, (11, 11), 5)

    # Convert to HSV
    hsv = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV_FULL)

    # Extract Green-Color Dinosaur
    mask = cv2.inRange(hsv, np.array([hl, sl, vl]), np.array([hu, su, vu]))
    binary_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Convert to binary
    # binary_img = cv2.bitwise_and(img, img, mask=mask)

    # Extract Level 1 Contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Return
    return (mask, contours) #(binary_img, contours)


def find_dinos(img, contours):
    """
    Uses morphological processing to detect dinosaurs from input binary image.

    :param img_pre_dinos: binary image
    :return: 2-tuple of (contour index, template index)

    @remark: Use global templates.
    """
    SIMILARITY_THRESHOLD = 0.3
    MIN_AREA_THRESHOLD = 3000

    # Select contours that are sufficiently large
    areas = [cv2.contourArea(cntr) for cntr in contours]
    minarea_idx = [i for i in range(len(areas)) if areas[i] > MIN_AREA_THRESHOLD]

    # For each contour, compare it with each of templates
    dino_contours = list()
    for idx in minarea_idx:
        cntr = contours[idx]

        # Compute match index for each template
        match_dino1 = cv2.matchShapes(cntr, dino1_template[0], 1, 0)    # using I1 moments
        match_dino2 = cv2.matchShapes(cntr, dino2_template[0], 1, 0)    # using I1 moments
        match_dino3 = cv2.matchShapes(cntr, dino3_template[0], 1, 0)    # using I1 moments
        match_dino4 = cv2.matchShapes(cntr, dino4_template[0], 1, 0)    # using I1 moments
        matches = [match_dino1, match_dino2, match_dino3, match_dino4]

        # If we have at least one good match, then choose the corresponding template
        if min(matches) < SIMILARITY_THRESHOLD:
            idx_template = matches.index(min(matches))
            idx_contour = idx
            dino_contours.append((idx_contour, idx_template))

    return dino_contours


def overlay_textures(img, contours, idx_template_pairs):
    """
    Applies the corresponding texture to the image.

    :param img: RGB image
    :param contours: contours object
    :param idx_template_pairs: 2-tuple of (contour index, template index)
    :return: texture overlayed image
    """
    print idx_template_pairs
    ret_img = img.copy()
    for idx_cntr, idx_tmp in idx_template_pairs:
        # Create mask
        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, contours, idx_cntr, (255, 255, 255), -1)

        # Resize the texture appropriately
        rows, cols = img.shape[0:2]
        txtr = cv2.resize(textures[idx_tmp], (cols, rows))

        # Apply mask over texture
        masked_img = cv2.bitwise_and(txtr, mask)
        _, binary_mask = cv2.threshold(masked_img, 1, 255, cv2.THRESH_BINARY)

        # Add the mask with original image1
        binary_mask = cv2.bitwise_not(binary_mask)
        crop_img = cv2.bitwise_and(ret_img, binary_mask)
        ret_img = crop_img + masked_img

    return ret_img


def main_test():
    # Load Templates
    if not load_templates():
        print 'Dinosaur Templates NOT Loaded.'
        return False

    print 'Dinosaur Templates Loaded.'

    # Load Textures
    if not load_textures():
        print 'Dinosaur Textures NOT Loaded.'
        return False

    print 'Dinosaur Textures Loaded.'

    # Setup Video Processing
    video = cv2.VideoCapture(0)

    while True:

        # Read Image
        _, img = video.read()
        r = _segment_color(img, RED, GAUSSIAN)
        y = _segment_color(img, YELLOW, GAUSSIAN)
        b = _segment_color(img, BLUE, GAUSSIAN)

        img, points = preprocess_homography(img)
        points = _filt_hgrf_points(points)
        #print np.array(points)
        tr_img = applyTransform(img, points)

        # Classify Dinosaur
        img_bin_dinos, contours_dinos = preprocess_dinos(tr_img)
        dinos = find_dinos(img_bin_dinos, contours_dinos)
        print len(dinos)

        # If NO dinosaur then
        if len(dinos) == 0:
            # Print on screen
            cv2.putText(img, 'No Dinosaur!', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            disp_img = img

        # Else if dinosaur(s) is (are) identified, then
        else:
            # Get Dino Pose(s)
            # dino_types, dino_poses = get_poses(img_bin_dinos, dinos)

            # Overlay texture(s) on corresponding dino(s)
            disp_img = overlay_textures(tr_img, contours_dinos, dinos)


        cv2.imshow('img', img)
        cv2.imshow('disp_img', disp_img)
        cv2.imshow('tr_img', tr_img)
        cv2.imshow('img_bin_dinos', img_bin_dinos)

        cv2.waitKey(3)
        if cv2.waitKey(3) == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main_test()

