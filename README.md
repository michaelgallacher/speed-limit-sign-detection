# speed-limit-sign-detection
[![CircleCI](https://circleci.com/gh/michaelgallacher/speed-limit-sign-detection.svg?style=shield)](https://circleci.com/gh/michaelgallacher/speed-limit-sign-detection)

## Overview
I was driving one afternoon when I noticed my phone notifying me I was over the posted speed limit.  I got curious about how hard it would be gather this information on a large scale.

I got a dash camera and recorded ~50 miles of local driving.  I anticipated having to go through tens of thousands of images, producing a dataset which I would then run through Yolo5.  

I didn't really want to annotate the videos by hand, so I created a small automated annotator.  The annotater turned out to be fast and rather accurate: ~73%

From there, I got the annotator running on a Raspberry PI device and was able to successfully identify and classify several signs in real time.
## Process
1. I ran the analyzer on ~300 videos, generating ~2000 images with alleged speed limit signs.
2. I removed ~500 images which were erroneous.
3. I used a great app (PhotoSweeper) to compare all the images and removed ~600 images I considered close enough to be duplicates.
4. I used Yolo5 (with the SpeedLimit.yaml) to train on ~900 images, which data are found in the 'images' and 'labels' directories.
## Usage
The following command will analyze all videos in the example folder. It will create two sets of files:
* frames containing the detected sign (the 'images' subdirectory)
* associated YOLO labels for the frames (the 'labels' subdirectory)

From there, you manually remove incorrectly labeled images. The format of the file name can be helpful.  The format is: year_date_time_frame_speed_random.jpg.  If the speed is not a multiple of 5, it's probably not a speed limit sign.   
```
track.py -v ./example/*.mp4
```
