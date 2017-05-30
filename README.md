# P05VehicleDetection



## 1 & 2. Extraction of features from the image dataset
In order for proper classification of images into cars and non cars, features of the give image dataset are extracted. 

Shape features and color features were extracted using the following methods, Histogram of orientation features (HOG), binning color feature and color histogram features.

The code used for extraction of features from each single image is defined in the following functon in 3rd code block of ipython file.

``` python
single_img_features(image, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
```

We read in all the images of car and non-car from the dataset provided. An example of car and non car images are displayed below

![png][car_noCar.png]

The HOG features were tested for various color spaces and the result image is displayed below.

![alt text][HOG_RGB_Gray_HSV.png]
![alt text][HOG_LUV_HLS_YUV.png]

After a lot of trial and error LUV and RGB color spaces seemed to be giving the best results for training using LinearSVM classifier.
LUV seems to be performing a little better than RGB. The spatial size, orient, hog_channel were all tuned to give the best result for the available dataset. The final tuned parameters can be seen below. 

``` python
color_space = 'LUV' # Can be RGB, GRAY, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()
``` 

## 3. Classifier used for classification and parameters

Spatial histogram and hog features were extracted for classification of images between vehicle and non-vehicle. 

The extracted classification data combining vehicle and non-vehicle was first normalised using ``` StandardScaler() ``` function from scikitlearn (sklearn.preprocessing) function. 20 percent of the data is split into validation data and the remaining 80 percent is used for training the classifier. 

The classifier used was a ``` LinearSVC ``` classifier from sklearn.svm . Hinge loss funtion was used for liear classifier.

The code was implemented in 7th codeblock of the jupyter notebook file.

# Sliding window search for cars and non-cars

## 4. Region selection and applying sliding window
Sliding window techniq similar to the method presented in the course lectures was used. The following box sizes 32 X 32, 64 X 64, 96 X 96  and 128 X 128  with corresponding y start and y stop limits can be observed in the code below. It can be observed that smaller boxes were chosen for upper lines. This is so smaller cars would be detcted at top of the image. Bigger boxes are used for the area of the image close to the car. smaller scales are  used for area farther from our car. This is due to the fact that the farther the distance from the car the smaller the perspective size of the vehicle in a camera image. the y limit was chose to be 400 any images above were filtered (it was assumed only trees and sky was visible around that area which is not important for detecting cars).

All the boxed image areas are rescaled to 64 X 64  and normalised to fit the training data. 

``` python
windowLimit = [((32, 32),  [400, 464]),
               ((64, 64),  [400, 500]),
               ((96, 96),  [400, 500]),
               ((128, 128),[450, 600])#,
               #((256, 256),[380, 640])
               ]
``` 

same parameters for training(mentioned in the above section) are used also for classification.

It was observed that LUV and RGB color spaces performed well in my case. LUV color space was a little better than RGB.

The overlapping of images were set to 0.5 and 0.5 in both x and y directions. Increasing the overlap limit would make more images but would give better values. But at the cost of processing time. So we chose to remain at these values. 

An image of all the boxes where  cars are searched can be seen in the figure below.
![alt text][allBoxes.png]

All the box coordinates are passed through a function called ``` search_windows() ``` . This function is used to search the window and classify if the box image is car. If the box image is detected to be car then the box coordinates are saved for further processing.

The searching window function is defined in codeblock 8 in the jupyter notebook.

```
search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True)
```

A few example images after detecting the car boxes using the  ``` search_windows() ``` funciton can be seen below

![alt text][boximg_before_FP_ex2.png]

![alt text][boximg_before_FP_ex5.png]

## 5. Filtering False Positive areas in the image

It could be observed from the above images that a few areas of the image without a car are also detected as areas with car. In order to filter those false positives in the image we add the number of boxes per image and do heat thresholding. These methods are implemented as described in the course. Then single box are drawn around the heat thresholded values. This method helps in eliminating the false positives.

The heat images after adding the heat and thresholding can be observed in the images below

![alt text][Heat_Map_ex1.png]

![alt text][Heat_Map_ex2.png]


The final images after thresholding can be observed below

![alt text][after_FP_ex1.png]

![alt text][after_FP_ex2.png]

The code for hot boxes thresholding and drawing boxes can be seen in code block 9

The functions used are
```
add_heat(heatmap, boxes)
heat_threshold(heatmap, threshold)
draw_full_boxes(img, labels)

```

## 6. Increasing the reliability of the model with hard mining.

There were still a few false positives after performing the heat thresholding. In order to eliminate this issue a few frames of image were taken and all the detected images are converted into 64 X 64 and saved. All the false positive images from the mined images are selected and added to the non-vehicle images bataset. This increased the reliability of the video by reducing false positives to half. and the classifier test accuracy increased by around 2 percent from 0.944 to 0.962. 

The code for hardming can be observed in codeblock 10

The function blocks used for this are
```
hardMining_ImgData(image)
get64_64_imgs(img, allWindows, img_number)  ## convert imgs from any shape to 64 X 64
```
# Video pipeline and tracking cars in the video

## 7. Video pipeline 

The output video 'DetectedCars_out.mp4' is attached to the project submission. It can be observed from the video that both the white and black cars are tracked reasonably well throughout the entire video. Thought there were a few issues.

The video pipeline is implemented in codeblock 12. For the video pipeline the color space, HOG features, binning dimensions and so on are the same as during training the linearSVM classifier for classification. 

The video pipeline function is 
```
process_pipeline(image)
```

MoviePy is used for generating videos from the pipeline. It is implemented in codeblocks 14 and 15 or the last two codeblocks. 

## 8. Tracking cars and locking cars

Initially tracking was not implemented and it was observed that the detected boxes were jumping from small to lare. Although the cars are being detected. 

In order to avoid this issue a tracking algorithm was implemented to track the cars make sure its not a false positive in ``` process_pipeline(image) ``` function. 
The tracking function is implemented in codeblock 11. It is called as follows
```
lock_boxes(boxes, prev_list)
```
In this algorithm cars are locked if they are continuously detected for 5 consecutive frames and unlocked if they are not detected for 5 consecutive frames. The boxes are only displayed if they are locked.

# Problems faced and potential solutions

It can be observed that the black car is lost for a second in the middle of the video. This could be because of over filtering during heat map thresholding. There are still false positives which could be because of lack of data or better classification. 

Better tuning of Color / Spatial parameters could help in solving this problem. Hardmining more images could also help in reducing 


After implementing the algorithm for tracking the cars. It can be observed that there are more than 2 boxes sometimes for one car.  And the box could be moving in a smooth form instead of jumping. This could be elimiated by better implementation of the algorithm.


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


###Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:


###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

