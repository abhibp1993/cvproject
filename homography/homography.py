from __future__ import print_function

import cv2
import numpy as np

def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1


# RED: hl=0;sl=127;vl=63;hu=15;su=255;vu=255;
# GREEN: hl=120;sl=80;vl=40;hu=140;su=255;vu=200;
# BLUE: hl=160;sl=80;vl=40;hu=167;su=255;vu=200;
# YELLOW: hl=35;sl=100;vl=40;hu=55;su=255;vu=255;
color_parameters = [
    ('red', [(0, 127, 90, 10, 255, 255), (160, 127, 90, 180, 255, 255)], (0, 0, 255)),
    #('green', (120, 80, 40, 140, 255, 200), (0, 255, 0)),
    ('blue', [(153, 120, 40, 167, 255, 200)], (255, 0, 0)),
    ('yellow', [(35, 100, 40, 55, 255, 255)], (0, 255, 255)),
]


def process_frame(frame):
    # Our operations on the frame come here

    out = frame
    outlist = []
    points = {}
    for color_name, params, color in color_parameters:

        blurred = cv2.medianBlur(frame, 5)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV_FULL)

        mask = None
        for param in params:
            hl, sl, vl, hu, su, vu = param
            this_mask = cv2.inRange(hsv, np.array([hl, sl, vl]), np.array([hu, su, vu]))
            if mask is None:
                mask = this_mask
            else:
                mask = cv2.bitwise_or(mask, this_mask)

        mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)))
        frame_out = cv2.bitwise_and(frame, frame, mask=mask)
        mask_out = cv2.cvtColor(mask_opened, cv2.COLOR_GRAY2BGR)
        outlist.append(mask_out)

        # Detect contours
        contours, _ = cv2.findContours(mask_opened.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(mask_out, contours, -1, color, 3)

        contours_moments = [(c, cv2.moments(c)) for c in contours]

        # Sort by area -- the m00 moment
        contours_moments.sort(key=lambda cm: cm[1]["m00"], reverse=True)

        n = 2 if color_name == 'blue' else 1

        points[color_name] = []

        # Display the largest contour, or 2 contours for blue
        for c, M in contours_moments[:n]:
            if M["m00"] == 0:
                break


            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            points[color_name].append(np.array([cX, cY]))

            # draw the contour and center of the shape on the image
            cv2.circle(mask_out, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(mask_out, "center", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    try:
        red = points['red'][0]
        yellow = points['yellow'][0]
        # Find which blue point is the top left
        if np.dot(yellow - points['blue'][0], points['blue'][0] - red) > 0:
            # dot product of the vectors is positive if it's the top left point, negative otherwise
            [blue_topleft, blue_bottomright] = points['blue']
        else:
            [blue_bottomright, blue_topleft] = points['blue']
    except (IndexError, ValueError):
        return reduce(cv2.max, outlist)

    width, height = frame.shape[:2]

    src_points = np.array([blue_topleft, yellow, red, blue_bottomright], dtype=np.float32)
    dst_points = np.array([[0, 0], [0, width], [height, 0], [height, width]], dtype=np.float32)

    print(src_points)
    print(dst_points)

    assert src_points.shape == dst_points.shape

    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    output = cv2.warpPerspective(frame, perspective_matrix, (height, width))

    # cv2.imshow("mask", reduce(cv2.max, outlist))
    return output

# Set up camera
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval, frame = False, None

# Main loop
while rval:
    cv2.imshow("preview", process_frame(frame))
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
