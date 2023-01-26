import argparse
import os
import random
import re
import time

import cv2
import imutils
import pytesseract
from pqdm.processes import pqdm

# If True, display the video with ROI annotations.
SHOW_GUI = False

# If True and a sign is detected, save the image and annotation files.
YOLO_FORMAT = True

# If True, print additional diagnostics to stdout.
DEBUG = False

# If True, attempts to detect text in the entire image instead of 
# attempting to locate just the speed limit number.
DETECT_ALL_TEXT = True

def dprint(str):
    if DEBUG:
        print(str)

# Load the video.
def process_video(video_file_path):
    dprint(f'processing: {video_file_path}')
    camera = cv2.VideoCapture(video_file_path)
    orig_width, orig_height = int(camera.get(3)), int(camera.get(4))
    paused = False

    # Keep looping over frames until there's no more.
    while True:
        if SHOW_GUI:
            wait_key = cv2.waitKey(1)
            # If the 'q' key is pressed, stop the loop and exit
            if wait_key & 0xFF == ord("q"):
                break

            if wait_key & 0xFF == ord(" "):
                paused = not paused

            if paused:
                continue

        # Grab the current frame.
        (grabbed, frame) = camera.read()
        if not grabbed:
            break

        # Find ROIs using a general HSV range.
        hsv_bright = [(0, 0, 128), (179, 32, 248)]
        speed = analyze_frame(frame, hsv_bright, orig_height, orig_width, video_file_path, SHOW_GUI)

        # If nothing is found, adjust the HSV range to help low contrast photos.
        if speed == 0:
            hsv_dim = [(0, 0, 64), (179, 64, 128)]
            speed = analyze_frame(frame, hsv_dim, orig_height, orig_width, video_file_path, SHOW_GUI)

        if SHOW_GUI:
            cv2.putText(frame, str(speed), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (0, 0, 255), 3)
            cv2.imshow("Tracking", frame)
    
    if SHOW_GUI:
        while cv2.waitKey(5) & 0xFF != ord(" "):
            continue

        # Cleanup the camera and close any open windows.
        camera.release()
        cv2.destroyAllWindows()


def analyze_frame(frame, hsv_range, orig_height, orig_width, video_file_path, show_gui):
    # Crop the image since we just need the upper right third.
    frame = frame[0: int(orig_height * 0.66), int(orig_width * .66):int(orig_width)]
    (frame_height, frame_width, _) = frame.shape

    blur = cv2.GaussianBlur(frame, (5, 5), 0)

    # Threshold to the requested HSV range.
    thresh = cv2.inRange(cv2.cvtColor(blur, cv2.COLOR_BGR2HSV), hsv_range[0], hsv_range[1])

    # Find contours in the image.
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    dprint(f'cnts: {cnts}')

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        # Threshold the contour based on the area contour compared to a proportion of the full image size.
        pixel_count = frame_height * frame_width
        # Using HD images, this comes out to between 2500 and 25,000 pixels.
        if not pixel_count // 333 < area < pixel_count // 33:
            continue

        rect = cv2.boundingRect(cnt)
        _, _, br_w, br_h = rect
        # Compare contour area to the contour bounding rect to get a general
        # idea if the blob is convex.
        if area < (br_w * br_h * 0.8):
            dprint(f'skipping rect due to convexity: {rect}')
            continue

        # Threshold the proportions of the contour to ensure it is a reasonable
        # sized rectangle.
        if not (br_w * 1.1 < br_h < br_w * 2.2):            
            dprint(f'skipping rect due to inertia: {rect}')
            continue

        # Convert from x,y,w,h to lt,rb.
        roi = [[rect[0], rect[1]], [rect[0] + rect[2], rect[1] + rect[3]]]
        roi_img = frame[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
        if show_gui:
            cv2.rectangle(frame, roi[0], roi[1], (0, 255, 255), 4)
            cv2.imshow("roi",roi_img)

        start = time.thread_time()
        if DETECT_ALL_TEXT:
            # Tell tesseract to find all text.
            config = ("-l eng --oem 1 --psm 11")
            text_found = pytesseract.image_to_string(roi_img, config=config)
            text_found = re.sub('\\D', '', text_found)
        else:
            roi_img_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            _, roi_thresh = cv2.threshold(roi_img_gray, 128, 255, cv2.THRESH_BINARY_INV)
            
            # Find the two largest contours in the image.
            numbers = imutils.grab_contours(cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
            numbers = sorted(numbers, key=cv2.contourArea, reverse=True)

            # Continue if there are 0 or 1 contours found.
            if len(numbers) < 2:
                continue

            if show_gui:
                cv2.imshow('roi thresh', roi_thresh)

            # Find the largest contour which is less than half the area of the ROI.
            ca = cv2.contourArea(numbers[0])
            while ca > 0.5 * rect[2] * rect[3]:
                numbers = numbers[1:]
                ca = cv2.contourArea(numbers[0])

            if len(numbers) < 2:
                continue

            # Continue if the 2nd largest contour is less than 5% of the ROI.
            ca = cv2.contourArea(numbers[1])
            if ca < 0.05 * rect[2] * rect[3]:
                continue

            # Determine the left-most contour.
            if min(numbers[0][:,0,0]) < min(numbers[1][:,0,0]):
                left_roi = cv2.boundingRect(numbers[0])
                right_roi = cv2.boundingRect(numbers[1])
            else:
                left_roi = cv2.boundingRect(numbers[1])
                right_roi = cv2.boundingRect(numbers[0])

            dprint(left_roi, right_roi)
            left_roi = [[left_roi[0], left_roi[1]], [left_roi[0] + left_roi[2], left_roi[1] + left_roi[3]]]
            right_roi = [[right_roi[0], right_roi[1]], [right_roi[0] + right_roi[2], right_roi[1] + right_roi[3]]]

            # Crop with an arbitrary margin.
            margin = 6
            left_image = roi_thresh[left_roi[0][1]-margin:left_roi[1][1]+margin, left_roi[0][0]-1:left_roi[1][0]+margin]
            right_image = roi_thresh[right_roi[0][1]-margin:right_roi[1][1]+margin, right_roi[0][0]-1:right_roi[1][0]+margin]

            if show_gui:
                cv2.imshow('li', left_image)
                cv2.imshow('ri', right_image)
                # while cv2.waitKey(5) & 0xFF != ord(" "):
                #     continue

            config = ("-l eng --oem 1 --psm 10")

            # Process the left and right numbers individually.
            left_text_found = pytesseract.image_to_string(left_image, config=config)
            left_text_found = re.sub('\\D', '', left_text_found)

            right_text_found = pytesseract.image_to_string(right_image, config=config)
            right_text_found = re.sub('\\D', '', right_text_found)

            # Return the speed limit.
            text_found = left_text_found + right_text_found


        dprint(f'time: {time.thread_time() - start}')
        if text_found != '':
            dprint("text->" + text_found + "<-")
            speed = int(text_found)
            if speed % 5 == 0:
                if YOLO_FORMAT:
                    # Determine file names and paths.
                    video_path, video_filename = os.path.split(video_file_path)
                    # Take the existing video name and append the speed limit. 
                    video_filename_base = f'{os.path.splitext(video_filename)[0]}_{speed}_{str(random.randint(100000,999999))}'
                    video_output_path = os.path.join(video_path, 'images')
                    os.makedirs(video_output_path, exist_ok=True)

                    # Save the full image.
                    output_image_path = os.path.join(video_output_path, f'{video_filename_base}.jpg')
                    dprint(f'saving {output_image_path}')
                    cv2.imwrite(output_image_path, frame)

                    # Save the ROI information to a text file.
                    yolo_text = f'0 {(rect[0] + rect[2] / 2.0) / frame_width} {(rect[1] + rect[3] / 2.0) / frame_height} {rect[2] / frame_width} {rect[3] / frame_height}'
                    video_output_path = os.path.join(video_path, 'labels')
                    os.makedirs(video_output_path, exist_ok=True)
                    text_output_path = os.path.join(video_output_path, f'{video_filename_base}.txt')
                    dprint(f'saving {text_output_path}')
                    with open(text_output_path, 'w+') as file_out:
                        file_out.write(yolo_text)
                else:
                    pass
                    # timestamp = strftime("%d-%b-%Y-%H-%M-%S", localtime())
                    # video_file_name = os.path.split(video_file_path)[1]
                    # output_file_name = f'out/{video_file_name}-{timestamp}-{int(random()*100000)}-{text_found}.png'
                    # dprint(f'saving {output_file_name}')
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

    if len(video_paths) == 1:
        SHOW_GUI = True
        process_video(video_paths[0])
    else:
        pqdm(video_paths, process_video, n_jobs=10)
