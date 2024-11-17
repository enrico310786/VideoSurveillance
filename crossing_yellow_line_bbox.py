import cv2
import numpy as np
import time
import os
import skvideo.io
font = cv2.FONT_HERSHEY_DUPLEX
import torch
import argparse
from ultralytics import YOLO

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

def find_yellow_line_parameter(image):
    # Set the min and max yellow in the HSV space
    yellow_light = np.array([20, 140, 200], np.uint8)
    yellow_dark = np.array([35, 255, 255], np.uint8)

    # transform the frame to hsv color space
    frame_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # isolate the yellow line
    mask_yellow = cv2.inRange(frame_hsv, yellow_light, yellow_dark)
    kernel = np.ones((4, 4), "uint8")
    # Moprphological closure to fill the black spots in white regions
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
    # Moprphological opening to fill the white spots in black regions
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

    # find the edge of isolate yellow line
    edges_yellow = cv2.Canny(mask_yellow, 50, 150)

    # find slope and intercept of yellow line
    coefficient_list = find_m_and_q(edges_yellow)
    print("len(coefficient_list): ", len(coefficient_list))

    return coefficient_list[0]


if __name__ == '__main__':
    '''
    Extract the first frame of a video
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video_path', type=str, default="resources/videos/video_stazione.mp4")
    parser.add_argument('--output_video_path', type=str, default="resources/videos/video_stazione_yolo_bbox.mp4")
    opt = parser.parse_args()
    input_video_path = opt.input_video_path
    output_video_path = opt.output_video_path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # Load a pretrained YOLO11n model Bbox model
    model_bbox = YOLO("yolo11n.pt")
    model_bbox.to(device)

    cap = cv2.VideoCapture(input_video_path)

    # Loop through the video frames
    frames_list = []
    count = 0
    yellow_light = np.array([20, 140, 200], np.uint8)
    yellow_dark = np.array([35, 255, 255], np.uint8)
    m, q, x_v = None, None, None

    start_time = time.time()

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
          # on the firts frame I set the parameters of the yellow straight line
          if count==0:
             m, q, x_v = find_yellow_line_parameter(frame)

          # Run YOLO inference on the frame
          results = model_bbox.predict(frame, classes=[0], conf=0.5, device=device) # 0 is person
          image_pred = results[0] # 0 because has been processed just one frame
          boxes = image_pred.boxes

          # iter over boxes
          for box in boxes:

              x1 = int(box.xyxy[0][0])
              y1 = int(box.xyxy[0][1])
              x2 = int(box.xyxy[0][2])
              y2 = int(box.xyxy[0][3])
              coords = (x1, y1-10)
              text = "persona"

              # check if the point (x2, y2) is over the line
              if m is not None and q is not None:
                  is_over = False
                  if m == np.inf:
                      if x2 > x_v:
                          is_over = True
                  else:
                      x2_yellow = (y2 - q)/m
                      if  x2 > x2_yellow:
                          is_over = True

                  if is_over:
                      color = (0, 0, 255) # colors in BGR
                  else:
                      color = (0, 255, 0) # colors in BGR

                  thickness = 2
                  frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                  frame = cv2.putText(frame, text, coords, font, 0.7, color, 2)
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frames_list.append(frame)
          count += 1

        else:
            # Break the loop if the end of the video is reached
            break


    print("count: ", count)
    end_time =  time.time()
    print("Elaboration time: ", end_time-start_time)
    out_video = np.array(frames_list)
    out_video = out_video.astype(np.uint8)
    print("out_video.shape: ", out_video.shape)

    # skvideo lavora in rgb, quindi i frames devono essere in rgb
    skvideo.io.vwrite(output_video_path,
                      out_video,
                      inputdict={'-r': str(int(30)),})