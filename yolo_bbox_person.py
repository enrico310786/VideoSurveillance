from ultralytics import YOLO
import argparse
import cv2

if __name__ == '__main__':
    '''
    Apply bbox to detected persons
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="resources/images/frame_yellow_line_0.png")

    opt = parser.parse_args()
    image_path = opt.image_path

    # Load a pretrained YOLO11n model
    model = YOLO("yolo11n.pt")
    image = cv2.imread(image_path)

    results = model.predict(image)

    # check that person has index name=0
    print("results[0].names: ", results[0].names)

    # iter over results. If there is only one frame then results has only one component
    for image_pred in results:
        class_names = image_pred.names
        boxes = image_pred.boxes

        # iter over the detected boxes and select thos of the person if exists
        for box in boxes:
            if class_names[int(box.cls)] == "person":
                print("person")
                print("person bbox: ", box.xyxy)

        image_pred.plot()
    image_pred.show()
