##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for exttracting hog features is in function get_hog_features in code P5_Vehicles_Detection_Tracking.ipynb.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car](https://github.com/devksingh/P5_Vehicles_Detection_Tracking/blob/master/images/car_image.png)
![non car](https://github.com/devksingh/P5_Vehicles_Detection_Tracking/blob/master/images/non_car.png)

I explored different color spaces and other parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  Below is the example of gray images and hog images of the car and non car image

![car gray](https://github.com/devksingh/P5_Vehicles_Detection_Tracking/blob/master/images/car_gray.png) ![car hog] (https://github.com/devksingh/P5_Vehicles_Detection_Tracking/blob/master/images/hog-image.png)
![non car gray](https://github.com/devksingh/P5_Vehicles_Detection_Tracking/blob/master/images/non_car_gray.png) ![non car hog](https://github.com/devksingh/P5_Vehicles_Detection_Tracking/blob/master/images/non_car_hog.png)

####2. Explain how you settled on your final choice of HOG parameters.
I used car and non car images with various color spaces and different combination of parameters and finally decided to go ahead with gray color space , orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`..

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using with following parameters
1. Bin Spatial (RGB)
2. Color Histogram with 3 channels(L of LUV, S of HLS & V of HSV)
3. HOG feature of gray channel

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decides to use 3 layers of window for search, which will try to find cars at various perspective level with different window. Below are the layers. Apart from these layers whenever I find any car then from next frame I will search area aroud it (+-5 pixels). The normal layer search will pe performed only in 1 out of three frame for performance reason.
![layer1](https://github.com/devksingh/P5_Vehicles_Detection_Tracking/blob/master/images/Window11.png)
![layer2](https://github.com/devksingh/P5_Vehicles_Detection_Tracking/blob/master/images/Window21.png)
![layer3](https://github.com/devksingh/P5_Vehicles_Detection_Tracking/blob/master/images/Window31.png)

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I used following features to train the SVM and predict the class:
1. Bin Spatial (RGB)
2. Color Histogram with 3 channels(L of LUV, S of HLS & V of HSV)
3. HOG feature of gray channel

I have already explained how I chose HOG feature, For choosing best color histogram I plotted one vehicle images with HLS,LUV and HSV channel and it was very obvious to choose L of LUV, S of HLS & V of HSV. Please find the image with different color channels

![luv](https://github.com/devksingh/P5_Vehicles_Detection_Tracking/blob/master/images/ColorFeature_Image_LUV.png)
![hls](https://github.com/devksingh/P5_Vehicles_Detection_Tracking/blob/master/images/ColorFeature_Image_HLS.png)
![hsv](https://github.com/devksingh/P5_Vehicles_Detection_Tracking/blob/master/images/ColorFeature_Image_HSV.png)
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://github.com/devksingh/P5_Vehicles_Detection_Tracking/blob/master/project_video_result_final.mp4)
I have also combined advance lane detection with this project. It worked well with project_video and challenge_video. I have uploaded the findlanelines.py code as well.

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  The code for this feature is defined in functions find_vehicles(), find_labeled_boxes() & print_boxes()

### Here is the complete pipeline:

![pipeline1](https://github.com/devksingh/P5_Vehicles_Detection_Tracking/blob/master/images/pipeline1.png)

![pipeline2](https://github.com/devksingh/P5_Vehicles_Detection_Tracking/blob/master/images/pipeline2.png)

![heatmap](https://github.com/devksingh/P5_Vehicles_Detection_Tracking/blob/master/images/heatmap.png)

![finaloutput](https://github.com/devksingh/P5_Vehicles_Detection_Tracking/blob/master/images/pipeline_output.png)



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest problem I faced was performance of the program as the slide window search was taking too much time. Here is what I did.
1. Search the 2 layers of window described above only on third frame
2. If possible car found then check only positions near to that car position for next 3 frames
3. If car is present in all the three frames then draw the box
4. Recalculate and update the box after six frame
This improved my performace and output because less wobbly.
