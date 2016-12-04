import copy
import cv2
import numpy as np

hl = 69
sl = 26
vl = 0
hu = 126
su = 255
vu = 255

cv2.namedWindow('image')

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))

cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    # Read frame
    ret, img = cap.read()

    # Filter Gaussian Noise
    blur = cv2.GaussianBlur(img, (11, 11), 5)

    # Convert to HSV and extract color
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV_FULL)
    mask = cv2.inRange(hsv, np.array([hl, sl, vl]), np.array([hu, su, vu]))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    img2 = cv2.bitwise_and(img, img, mask=mask)

    # Contour identification
    contours, hierarchy = cv2.findContours(opening, 1, 2)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    areas = [(areas[i], i) for i in range(len(areas)) if areas[i] > 5000]
    print areas

    # Fit convex hull
    if len(areas) > 0:
        areas = areas[0]
        cnt = contours[areas[1]]
        hull = cv2.convexHull(cnt)

    drawing = np.zeros(img.shape)
    for i in xrange(len(contours)):
        if cv2.contourArea(contours[i]) > 5000:  # just a condition
            cv2.drawContours(drawing, contours, i, (255, 255, 255), 1, 8, hierarchy)

    # Show Updates
    cv2.imshow("mask", drawing)
    cv2.imshow("erode", img)
    cv2.imshow("image", img2)

    if cv2.waitKey(3) == ord('q'):
        break

cv2.destroyAllWindows()

