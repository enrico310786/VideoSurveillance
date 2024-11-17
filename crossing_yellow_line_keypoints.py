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

def check_crossing_yellow_line(m, q, x_v, x, y):
    is_over = False

    if m != np.inf:
        x_yellow = (y-q)/m
        if x > x_yellow:
            is_over = True
    else:
        if x > x_v:
            is_over = True

    return is_over


if __name__ == '__main__':
    '''
    Extract the first frame of a video
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video_path', type=str, default="resources/videos/video_stazione.mp4")
    parser.add_argument('--output_video_path', type=str, default="resources/videos/video_stazione_yolo_keypoints.mp4")
    opt = parser.parse_args()
    input_video_path = opt.input_video_path
    output_video_path = opt.output_video_path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # Load a pretrained YOLO11n model Bbox model
    model_keypoints = YOLO("yolo11m-pose.pt")
    model_keypoints.to(device)

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
          #results = model_keypoints(image)  # results list
          results = model_keypoints.predict(frame, device=device)
          keypoints = results[0].keypoints
          xy = keypoints.xy
          print("Detected person: ", xy.shape[0])

          # iter over persons
          for idx_person in range(xy.shape[0]):
              # iter over keypoints of a fixed person to find the bbox and the 15 and 16 keypoints
              list_x = []
              list_y = []

              x15 = None
              y15 = None
              x16 = None
              y16 = None

              for i, th in enumerate(xy[idx_person]):
                  x = int(th[0])
                  y = int(th[1])

                  if x != 0.0 and y != 0.0:
                      list_x.append(x)
                      list_y.append(y)

                      if i == 15:
                          x15 = x
                          y15 = y
                      elif i == 16:
                          x16 = x
                          y16 = y

              # check the crossing of the yellow line
              is_over = False
              if x15 is not None and y15 is not None:
                is_over = check_crossing_yellow_line(m, q, x_v, x15, y15)
              if not is_over and x16 is not None and y16 is not None:
                  is_over = check_crossing_yellow_line(m, q, x_v, x16, y16)

              # build the bbox of the person if the lists are not empty
              if len(list_x) > 0 and len(list_y) > 0:
                  min_x = min(list_x)
                  max_x = max(list_x)
                  min_y = min(list_y)
                  max_y = max(list_y)

                  w = max_x - min_x
                  h = max_y - min_y
                  dx = int(w / 3)
                  x0 = min_x - dx
                  x1 = max_x + dx
                  y0 = min_y - dx
                  y1 = max_y + dx

                  coords = (x0, y0 - 10)
                  text = "person"

                  if is_over:
                      color = (0, 0, 255)  # colors in BGR
                  else:
                      color = (0, 255, 0)  # colors in BGR
                  thickness = 3

                  frame = cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)
                  frame = cv2.putText(frame, text, coords, font, 0.7, color, 2)
                  print("idx_person: {} - is_over: {}".format(idx_person, is_over))

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