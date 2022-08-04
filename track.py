import argparse
import os
import random
import re

import cv2
import imutils
import pytesseract
from pqdm.processes import pqdm

show_gui = False

yolo_format = True


# load the video
def process_video(video_file_path):
    print(f'procesing: {video_file_path}')
    camera = cv2.VideoCapture(video_file_path)
    orig_width, orig_height = int(camera.get(3)), int(camera.get(4))
    paused = False

    # keep looping over frames until there's no more
    while True:
        if show_gui:
            wait_key = cv2.waitKey(1)
            # if the 'q' key is pressed, stop the loop
            if wait_key & 0xFF == ord("q"):
                break

            if wait_key & 0xFF == ord(" "):
                paused = not paused

            if paused:
                continue

        # grab the current frame
        (grabbed, frame) = camera.read()
        if not grabbed:
            break

        ysv_bright = [(0, 0, 128), (179, 32, 248)]
        speed = analyze_frame(frame, ysv_bright, orig_height, orig_width, video_file_path, show_gui)

        if speed == 0:
            ysv_dim = [(0, 0, 64), (179, 64, 128)]
            speed = analyze_frame(frame, ysv_dim, orig_height, orig_width, video_file_path, show_gui)

        if show_gui:
            cv2.putText(frame, str(speed), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (0, 0, 255), 3)
            cv2.imshow("Tracking", frame)
    
    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()


def analyze_frame(frame, ysv_range, orig_height, orig_width, video_file_path, show_gui):
    frame = frame[0: int(orig_height * 0.66), int(orig_width * .66):int(orig_width)]
    (frame_height, frame_width, _) = frame.shape
    # frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    blur = cv2.GaussianBlur(frame, (5, 5), 0)

    gray = cv2.inRange(cv2.cvtColor(blur, cv2.COLOR_BGR2HSV), ysv_range[0], ysv_range[1])
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    # blur = cv2.morphologyEx(blur, cv2.MORPH_ERODE, element)
    thresh = gray
    # _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, element)
    # find contours in the image
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # print(f'cnts: {cnts}')

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if not 2500 < area < 25000:
            continue

        if True:
            rect = cv2.boundingRect(cnt)
            _, _, br_w, br_h = rect
            if area < (br_w * br_h * 0.8):
                # print(f'skipping rect due to convexity: {rect}')
                continue

        if not (br_w * 1.1 < br_h < br_w * 2.2):
            # print(f'skipping rect due to inertia: {rect}')
            continue

        # convert from x,y,w,h to lt,rb
        margin = 0
        roi = [[rect[0], rect[1]], [rect[0] + rect[2], rect[1] + rect[3]]]
        roi_img = frame[roi[0][1] - margin:roi[1][1] + margin, roi[0][0] - margin:roi[1][0] + margin].copy()
        if show_gui:
            cv2.rectangle(frame, roi[0], roi[1], (0, 255, 255), 4)
            cv2.imshow("Thres", frame[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]])

        config = ("-l eng --oem 1 --psm 11")
        text_found = pytesseract.image_to_string(roi_img, config=config)
        text_found = re.sub('\\D', '', text_found)

        if text_found != '':
            # print("text->" + text_found + "<-")
            speed = int(text_found)
            if speed % 5 == 0:
                if yolo_format:
                    video_path, video_filename = os.path.split(video_file_path)
                    video_filename_base = f'{os.path.splitext(video_filename)[0]}_{speed}_{str(random.randint(100000,999999))}'
                    video_output_path = os.path.join(video_path, 'output')
                    os.makedirs(video_output_path, exist_ok=True)

                    output_image_path = os.path.join(video_output_path, f'{video_filename_base}.jpg')
                    print(f'saving {output_image_path}')
                    cv2.imwrite(output_image_path, frame)

                    yolo_text = f'0 {(rect[0] + rect[2] / 2.0) / frame_width} {(rect[1] + rect[3] / 2.0) / frame_height} {rect[2] / frame_width} {rect[3] / frame_height}'
                    text_output_path = os.path.join(video_output_path, f'{video_filename_base}.txt')
                    # print(f'saving {text_output_path}')
                    with open(text_output_path, 'w+') as file_out:
                        file_out.write(yolo_text)
                else:
                    pass
                    # timestamp = strftime("%d-%b-%Y-%H-%M-%S", localtime())
                    # video_file_name = os.path.split(video_file_path)[1]
                    # output_file_name = f'out/{video_file_name}-{timestamp}-{int(random()*100000)}-{text_found}.png'
                    # print(f'saving {output_file_name}')
                    # margin = 1
                    # roi_out = frame[roi[0][1]-margin:roi[1][1]+margin, roi[0][0]-margin:roi[1][0]+margin].copy()
                    # cv2.imwrite(output_file_name, roi_out)
            return speed

    return 0


if __name__ == '__main__':
    os.makedirs('out', exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", nargs='+', help="path to the (optional) video file")
    args = vars(ap.parse_args())
    video_paths = args["video"]

    show_gui = len(video_paths) == 1

    # Parallel(n_jobs=-1)(delayed(process_video)(video_file_path) for video_file_path in tqdm.tqdm(video_paths))
    # process_video(video_paths[0])
    pqdm(video_paths, process_video, n_jobs=10)
