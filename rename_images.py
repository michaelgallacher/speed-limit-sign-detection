import argparse
from hashlib import new
from os import path, rename
import random
import re
import glob
import cv2
import imutils
import pytesseract
from pqdm.processes import pqdm


if __name__ == '__main__':
    img_files = glob.glob('images/*.jpg')
    count = 1
    for img_file in img_files:
        original_img_path, original_img_filename = path.split(img_file)

        original_img_filename, original_img_extension = path.splitext(original_img_filename)

        new_filename = str(count).zfill(5)

        new_image_filename = path.join(original_img_path, new_filename + original_img_extension)
        rename(img_file, new_image_filename)

        # handle txt files
        original_txt_file = path.join(path.split(original_img_path)[0], 'labels', original_img_filename + '.txt')
        new_txt_filename = path.join(path.split(original_img_path)[0], 'labels', new_filename + '.txt')
        rename(original_txt_file, new_txt_filename)
        count += 1