import cv2
import numpy as np


def find_slope(x1, y1, x2, y2):
    if x2 != x1:
        return ((y2 - y1) / (x2 - x1))
    else:
        return np.inf


def find_m_and_q(edges):

    lines = cv2.HoughLinesP(
            edges, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=120, # Min number of votes for valid line
            minLineLength=80, # Min allowed length of line
            maxLineGap=10 # Max allowed gap between line for joining them
            )

    coefficient_list = []

    # Iterate over points
    if lines is None:
        # if line is None return an empty list of coefficients
        return coefficient_list
    else:
        for points in lines:
            x_vertical = None

            # Extracted points nested in the list
            x1,y1,x2,y2=points[0]

            slope = find_slope(x1, y1, x2, y2)
            if slope == np.inf:
                # if the slope is infinity the intercept is None and set the x vertical
                intercept = None
                x_vertical = x1
            else:
                intercept = y1-(x1*(y2-y1)/(x2-x1))

            coefficient_list.append((slope, intercept, x_vertical))
    print("coefficient_list: ", coefficient_list)
    return coefficient_list


def draw_lines(image, list_coefficient):
    image_line = image.copy()
    h, w = image_line.shape[:2]
    for coeff in list_coefficient:
        m, q, x_v = coeff
        y0 = 0
        y1 = h
        if m != np.inf:
            x0 = -q/m
            x1 = (h-q)/m
        else:
            x0 = x1 = x_v
        cv2.line(image_line, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 6)
    return image_line

###########################################################################
###########################################################################

# Load the image
frame_path = '/home/enrico/Projects/VideoSurveillance/resources/images/frame_yellow_line.png'
image = cv2.imread(frame_path)

# Set the min and max yellow in the HSV space
yellow_light=np.array([20,140,200],np.uint8)
yellow_dark=np.array([35,255,255],np.uint8)

# transform the frame to hsv color space
frame_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
frame_hsv_path = "/home/enrico/Projects/VideoSurveillance/resources/images/hsv.png"
cv2.imwrite(frame_hsv_path, frame_hsv)

# isolate the yellow line
mask_yellow=cv2.inRange(frame_hsv,yellow_light,yellow_dark)
kernel=np.ones((4,4),"uint8")
# Moprphological closure to fill the black spots in white regions
mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
# Moprphological opening to fill the white spots in black regions
mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
mask_yellow_path = "/home/enrico/Projects/VideoSurveillance/resources/images/mask_yellow.png"
cv2.imwrite(mask_yellow_path, mask_yellow)

# find the edge of isolate yellow line
edges_yellow = cv2.Canny(mask_yellow,50,150)
edges_yellow_path = "/home/enrico/Projects/VideoSurveillance/resources/images/edges_yellow.png"
cv2.imwrite(edges_yellow_path, edges_yellow)

# find slope and intercept of yellow line
coefficient_list = find_m_and_q(edges_yellow)
print(len(coefficient_list))

# drow lines
image_lines = draw_lines(image, coefficient_list)
image_lines_path = "/home/enrico/Projects/VideoSurveillance/resources/images/image_lines.png"
cv2.imwrite(image_lines_path, image_lines)