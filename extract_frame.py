import cv2
import os
import argparse

def extract_frames(video_path, output_folder, index_frame):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    count_frame = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break

        if count_frame==index_frame:
            frame_path = os.path.join(output_folder, "frame_yellow_line_" + str(index_frame) + ".png")
            cv2.imwrite(frame_path, frame)  # Save the frame
            print("extracted frame: ", index_frame)
            break

        count_frame += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    '''
    Extract the first frame of a video
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default="/home/enrico/Projects/VideoSurveillance/resources/videos/video_stazione.mp4")
    parser.add_argument('--output_folder', type=str, default="/home/enrico/Projects/VideoSurveillance/resources/images")
    parser.add_argument('--index_frame', type=int, default=0)

    opt = parser.parse_args()
    video_path = opt.video_path
    output_folder = opt.output_folder
    index_frame = int(opt.index_frame)
    print("index_frame: ", index_frame)

    extract_frames(video_path, output_folder, index_frame)