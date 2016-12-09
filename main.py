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

##template = cv2.imread('img/4.bmp', 0)
##print len(template[np.where(template>200)])
##ret,thresh = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
##contours1, hierarchy1 = cv2.findContours(thresh, 1, 2)
##
##print len(contours1), [cv2.contourArea(cnt) for cnt in contours1]

##template = open('dino4.txt', 'r')
##template.
##template = template.split(',')[:-1]
##template = np.array(template)
##template.reshape((640, 480))
##cv2.imshow('img', template)
##cv2.waitKey(0)
try:
    template1 = np.loadtxt('dino1.txt',np.uint8)
    contours1, hierarchy1 = cv2.findContours(template1, 1, 2)
    template2 = np.loadtxt('dino2.txt', np.uint8)
    contours2, hierarchy2 = cv2.findContours(template2, 1, 2)
    template3 = np.loadtxt('dino3.txt',np.uint8)
    contours3, hierarchy3 = cv2.findContours(template3, 1, 2)
    template4 = np.loadtxt('dino4.txt',np.uint8)
    contours4, hierarchy4 = cv2.findContours(template4, 1, 2)
    #areas = [cv2.contourArea(cnt) for cnt in contours1]
    #print (areas)
    #template = np.fromarray(template)
    #print len(contours1)
    #ret, thresh = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
    
    cv2.imshow('hello', template1)
except  Exception, e:
    print e

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
    #print areas

    # Fit convex hull
    if len(areas) > 0:
        areas = areas[0]
        cnt = contours[areas[1]]
        hull = cv2.convexHull(cnt)

    drawing = np.zeros(img.shape[0:2])
    for i in xrange(len(contours)):
        if cv2.contourArea(contours[i]) > 5000:  # just a condition
            cv2.drawContours(drawing, contours, i, 255, 1, 8, hierarchy)

    for c in contours:
        ret = cv2.matchShapes(c, contours2[0], 1, 0.0)
        print(ret)
    # Show Updates
    cv2.imshow("mask", drawing)
##    cv2.imshow("erode", thresh)
    cv2.imshow("image", img2)

    if cv2.waitKey(3) == ord('p'):
##        f = open('dino4.txt', 'w')
##        x, y = drawing.shape
##        for i in range(x):
##            for j in range(y):
##                if drawing[i, j] != 0:
##                    f.write(str(255) + ',')
##                else:
##                    f.write(str(0) + ',')
##        f.close()
##        
         np.savetxt('dino4.txt', drawing)

         
    if cv2.waitKey(3) == ord('q'):
        break

cv2.destroyAllWindows()

