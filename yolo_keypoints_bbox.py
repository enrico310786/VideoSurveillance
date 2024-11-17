from ultralytics import YOLO
import cv2
font = cv2.FONT_HERSHEY_DUPLEX


# Load a pretrained YOLO11n-pose Pose model
model = YOLO("yolo11m-pose.pt")

# Run inference on an image
path_image = "resources/images/frame_yellow_line_900.png"
image = cv2.imread(path_image)
annotated_frame_keypoints = image.copy()
annotated_frame_bbox = image.copy()
results = model(image)  # results list

# extract keypoints
keypoints = results[0].keypoints
conf = keypoints.conf
xy = keypoints.xy
print(xy.shape)  # (N, K, 2) where N is the number of person detected
print("Detected person: ", xy.shape[0])

# iter over persons
for idx_person in range(xy.shape[0]):

    print("idx_person: ", idx_person)

    #iter over keypoints of a fixed person
    list_x = []
    list_y = []
    for i, th in enumerate(xy[idx_person]):
        x = int(th[0])
        y = int(th[1])

        if x !=0.0 and y!=0.0:

            list_x.append(x)
            list_y.append(y)
            print("x: {} - y: {}".format(x, y))
            annotated_frame_keypoints = cv2.circle(annotated_frame_keypoints, (x,y), radius=3, color=(0, 0, 255), thickness=-1)
            annotated_frame_keypoints = cv2.putText(annotated_frame_keypoints, str(i), (x, y-5), font, 0.7, (0, 0, 255), 2)

    if len(list_x) > 0 and len(list_y) > 0:
        min_x = min(list_x)
        max_x = max(list_x)
        min_y = min(list_y)
        max_y = max(list_y)
        print("min_x: {} - max_x: {} - min_y: {} - max_y: {}".format(min_x, max_x, min_y, max_y))
        w = max_x - min_x
        h = max_y - min_y
        dx = int(w/3)
        x0 = min_x - dx
        x1 = max_x + dx
        y0 = min_y - dx
        y1 = max_y + dx
        print("x0: {} - x1: {} - y0: {} - y1: {}".format(x0, x1, y0, y1))

        coords = (x0, y0 - 10)
        text = "person"
        color = (0, 255, 0) # colors in BGR
        thickness = 3
        annotated_frame_bbox = cv2.rectangle(annotated_frame_bbox, (x0, y0), (x1, y1), color, thickness)
        annotated_frame_bbox = cv2.putText(annotated_frame_bbox, text, coords, font, 0.7, color, 2)


annotated_frame_path = "/home/enrico/Projects/VideoSurveillance/resources/images/annotated_frame_keypoints_900.png"
cv2.imwrite(annotated_frame_path, annotated_frame_keypoints)

annotated_frame_bbox_path = "/home/enrico/Projects/VideoSurveillance/resources/images/annotated_frame_keypoints_bbox_900.png"
cv2.imwrite(annotated_frame_bbox_path, annotated_frame_bbox)