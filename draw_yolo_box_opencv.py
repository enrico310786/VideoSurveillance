from ultralytics import YOLO
import cv2
font = cv2.FONT_HERSHEY_DUPLEX
import argparse


if __name__ == '__main__':
    '''
    Apply bbox to detected persons
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="resources/images/frame_yellow_line_900.png")
    parser.add_argument('--annotated_frame_path', type=str, default="resources/images/annotated_frame_900.png")
    opt = parser.parse_args()
    image_path = opt.image_path
    annotated_frame_path = opt.annotated_frame_path

    # Load a pretrained YOLO11n model
    model = YOLO("yolo11n.pt")
    image = cv2.imread(image_path)
    annotated_frame = image.copy()

    # set in the predict function the interested classes to detect. Here I want to detect persons, whose index is 0
    results = model.predict(image, classes=[0], conf=0.54)
    image_pred = results[0]
    boxes = image_pred.boxes

    # iter over all the detected boxes of persons
    for box in boxes:

        x1 = int(box.xyxy[0][0])
        y1 = int(box.xyxy[0][1])
        x2 = int(box.xyxy[0][2])
        y2 = int(box.xyxy[0][3])
        coords = (x1, y1 - 10)
        text = "person"
        print("x1: {} - y1: {} - x2: {} - y2: {}".format(x1, y1, x2, y2))

        color = (0, 255, 0) # colors in BGR
        thickness = 3
        annotated_frame = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        annotated_frame = cv2.putText(annotated_frame, text, coords, font, 0.7, color, 2)

    # save the image with the bounding boxes drown using opencv
    cv2.imwrite(annotated_frame_path, annotated_frame)