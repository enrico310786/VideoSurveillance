# VideoSurveillance


This Computer Vision project intends to determine the occurrence of dangerous situations that can put people's lives at risk.
At the moment is analysed the case of crossing the yellow line of a station while waiting for trains.

The control is carried out using different technologies such as:

- YOLO11 for the person detection
- Hough transform for the yellow line detection and the determination of the corresponding straight line equation

## YOLO11 person detection: bounding box

Applying the following script I use the model "yolo11n.pt" to detect the persons on one image and drow
the corresponding bounding box

```bash
python yolo_bbox_person.py --image_path <path to the image with persons>
```


## YOLO11 person detection: drow bounding box with opencv

Applying the following script I use the model "yolo11n.pt" to detect the persons on one image. To drow the bounding box I
I use the coordinates of each box and drow the connection lines using opencv

```bash
python draw_yolo_box_opencv.py --image_path <path to the image with persons> --annotated_frame_path <path where to save the image with bbox>
```


## YOLO11 pose detection: keypoints

Applying the following script I use the model "yolo11m-pose.pt" to detect the pose of each person on one image.
Using the coordinates of the keypoints I also determine the bounding box around each person

```bash
python yolo_keypoints_bbox.py --image_path <path to the image with persons> --annotated_frame_path <path where to save the image with keypoints> --annotated_frame_bbox_path <path where to save the image with bounding boxes>
```

## Yellow line equation by Hough transform