# speed-limit-sign-detection

I was driving one afternoon when I noticed my phone notifying me I was over the posted speed limit.  I got curious about how hard it would be gather this information on a large scale.

I got a dash camera and recorded ~50 miles of local driving.  I anticipated having to go through tens of thousands of images, producing a dataset which I would then run through Yolo5.  

I didn't really want to annotate the video by hand, so I created a small automated annotator.  The annotater turned out to be fast and rather accurate.

I then got the annotator running on a Raspberry PI device and was able to successfully identify and read a sign while driving!

This repo is the result.