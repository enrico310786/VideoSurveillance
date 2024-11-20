# VideoSurveillance


This Computer Vision project intends to determine the occurrence of dangerous situations that can put people's lives at risk.
At the moment is analysed the case of crossing the yellow line of a station while waiting for trains.

The control is carried out using different technologies such as:

- YOLO11 for the person detection
- Hough transform for the yellow line detection and the determination of the corresponding straight line equation

## Installation

Build the environment based on python3.10

```bash
python3.10 -m venv venv
```

install all the packages using
```bash
pip install -r requirements.txt
```


## YOLO11 person detection: bounding box

With the following script I can use the model "yolo11n.pt" to detect the persons on one image and drow
the corresponding bounding box

```bash
python yolo_bbox_person.py --image_path <path to the image with persons>
```


## YOLO11 person detection: drow bounding box with opencv

With the following script I can use the model "yolo11n.pt" to detect the persons on one image. To drow the bounding box I
I use the coordinates of each box and drow the connection lines using opencv

```bash
python draw_yolo_box_opencv.py --image_path <path to the image with persons> --annotated_frame_path <path where to save the image with bbox>
```


## YOLO11 pose detection: keypoints

With the following script I can use the model "yolo11m-pose.pt" to detect the pose of each person on one image.
Using the coordinates of the keypoints I also determine the bounding box around each person

```bash
python yolo_keypoints_bbox.py --image_path <path to the image with persons> --annotated_frame_path <path where to save the image with keypoints> --annotated_frame_bbox_path <path where to save the image with bounding boxes>
```

## Yellow line equation by Hough transform

Given an image with the yellow line near the track at the station, with the following script I can
detect the yellow line and obtain the straight line equation using the Hough transform by openCV

```bash
python find_yellow_line_equation.py --frame_path <path to the image with yellow line> --frame_hsv_path <path where to save the image in hsv format> --mask_yellow_path <path where to save the detected mask of the yellow line> --edges_yellow_path <path where to save the image of the edges of the yellow line> --image_lines_path <path where to save the original image with the straight line corresponding to the yellow line>
```

## Yellow line crossing using YOLO11 bounding box

The first strategy to monitor when a person crosses the yellow line is to check if the low-right vertex of
the corresponing bounding box is over the yellow line. With the following script I can detect all the person 
on each frame of the video and drow a bounding box colored with green if the person is not crossing the yellow line
and with red otherwise

```bash
python crossing_yellow_line_bbox.py --input_video_path <path to the original video> --output_video_path <path to the video with the colored bounding boxes for the detected person>
```

## Yellow line crossing using YOLO11 keypoints

The second strategy to monitor when a person crosses the yellow line is based on the use of the 
feet keypoints. When one or both of the feet keypoints are beyond the yellow line, 
it can be concluded that the person has crossed the line. I apply this strategy using the following script

```bash
python crossing_yellow_line_keypoints.py --input_video_path <path to the original video> --output_video_path <path to the video with the colored bounding boxes for the detected person>
```