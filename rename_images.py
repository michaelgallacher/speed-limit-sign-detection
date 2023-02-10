from os import path, rename
import glob

# Takes a set of images and renames them (and their associated label file) to
# a simple numbering scheme starting with 00001.jpg
if __name__ == '__main__':
    img_files = glob.glob('images/*.jpg')
    count = 1
    for img_file in img_files:
        # Split the path and the file name.
        original_img_path, original_img_filename = path.split(img_file)

        original_img_filename, original_img_extension = path.splitext(original_img_filename)

        # Create a new file name.
        new_filename = str(count).zfill(5)

        # Rename the image file.
        new_image_filename = path.join(original_img_path, new_filename + original_img_extension)
        rename(img_file, new_image_filename)

        # Find the annotation with the same file name and rename it to the same new number.
        original_txt_file = path.join(path.split(original_img_path)[0], 'labels', original_img_filename + '.txt')
        new_txt_filename = path.join(path.split(original_img_path)[0], 'labels', new_filename + '.txt')
        rename(original_txt_file, new_txt_filename)
        count += 1
