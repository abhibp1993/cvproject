#!/usr/bin/env python
"""
Computer Vision Project
Title: ??

Authors: Abhishek N. Kulkarni, Shravan Murlidaran, John Pryor
"""

# Imports
import cv2
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


def preprocess_homography(img):
    """
    Preprocesses image to a binary images to detect color-coded end points.

    :param img: Raw RGB image
    :return: 4 corner points as 2-tuples in image coordinate [bottom-left, bottom-right, top-right, top-left]
    """

    # Filter Image
    blur_img = cv2.GaussianBlur(img, (11, 11), 5)

    # Convert to HSV
    hsv = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV_FULL)

    # Extract Red-Color Dinosaur
    red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([179, 255, 255]))
    mask = cv2.bitwise_or(red_mask1, red_mask2)

    
    return mask



def apply_homography(img, points):
    """
    Applies the homography transform on img using 4 reference points.

    :param img: RGB Image
    :param points: Points as 2-tuples in order [bottom-left, bottom-right, top-right, top-left]
    :return: 2-tuple of (transformed image, 2D plane pose)

    @remark: Assume knowledge of original points coordinates in real-world.
    """
    pass


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


def get_poses(img, dinos):
    """
    Estimates pose of each of the dinosaurs.

    :param img: binary image
    :param dinos: ??
    :return: pose of each dino in the image.
    """
    pass


def overlay_textures(img, contours, idx_template_pairs):
    """
    Applies the corresponding texture to the image.

    :param img: RGB image
    :param contours: contours object
    :param idx_template_pairs: 2-tuple of (contour index, template index)
    :return: texture overlayed image
    """
    print idx_template_pairs
    ret_img = img
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


def main():
    """
    Implements the processing loop.
    """
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

    # Loop
    while True:
        # Read Image
        ret, img = video.read()

        # Homography
        img_pre_hpoints = preprocess_homography(img)
        # hgraph_img, world_pose = apply_homography(img, img_pre_hpoints)

        # Classify Dinosaur
        img_bin_dinos, contours_dinos = preprocess_dinos(img)
        dinos = find_dinos(img_bin_dinos, contours_dinos)

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
            disp_img = overlay_textures(img, contours_dinos, dinos)

        # Show Image
        cv2.imshow("raw_input", img)
        cv2.imshow("dino_segmented", disp_img)
        cv2.imshow("red-color", img_pre_hpoints)
        cv2.waitKey(3)


if __name__ == '__main__':
    main()

