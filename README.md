## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image1]: ./output_images/Vehicle_Not_Vehicle.png
[image2]: ./output_images/Colorspace_HSV.png
[image3]: ./output_images/Colorspace_YCrCb.png
[image4]: ./output_images/Slide_Windows.png
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/heatmap_single.png
[image7]: ./output_images/heatmap_overlap.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Submission

* VehicleDetect_Final.ipynb: The jupyter note of vehicle detection program implemented in Python3.  
* Readme.ipynb: Documents the method and parameters used in the Vehicle Detection program. You're reading it!
* project_video_output.mp4: The output video with detected vehicles highlighted
* output_images: Folder contains all intermediate output images during development

---


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the Color and Gradient cell of the IPython notebook


I started by reading in all the `vehicle` and `non-vehicle` images. There are 8792 vehicle images and 8968 non-vehicle images, which are well balanced on amounts, this avoids the overfitting to one side of prediction. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image3]

Here is an example using the `HSV` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

Intuitively, YCrCb CH1 and HSV CH3 remained the features of the car. We would still need to verify that through more tests.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried multiple combinations of parameters and focused on the RGB, HSV and YCrCb colorscpaces. RGB as the baseline, HSV and YCrCb as targing color space. Tradeoff in parameters tuning is mainly focused on the accuracy and computational time on 10 predictions. A lower prediction time is more realistic in this real time processing scenario. A comparison of results are shown in the following table. 

| index | ColorSpace | ColorChannels | Test Accuracy | 10samples Prediction Time (s) |
|-------|------------|---------------|---------------|-------------------------------|
| 1     | RGB        | 1             | 0.9761        | 0.00216                       |
| 2     | RGB        | 2             | 0.9772        | 0.0022                        |
| 3     | RGB        | 3             | 0.9789        | 0.00205                       |
| 4     | RGB        | ALL           | 0.9778        | 0.00241                       |
| 5     | HSV        | 1             | 0.9682        | 0.00207                       |
| 6     | HSV        | 2             | 0.9727        | 0.0021                        |
| 7     | HSV        | 3             | 0.9823        | 0.00216                       |
| 8     | HSV        | ALL           | 0.9814        | 0.00627                       |
| 9     | YCrCb      | 1             | 0.9845        | 0.00218                       |
| 10    | YCrCb      | 2             | 0.9761        | 0.00206                       |
| 11    | YCrCb      | 3             | 0.9662        | 0.00211                       |
| 12    | YCrCb      | ALL           | 0.9859        | 0.01401                       |

Over the 12 runs, Channel 0 of YCrCb has the best performance over accuracy with reasonable prediction time.  Another interesting inspection is that, "All" channels don't necessarily represent high accuracy cuz more non-characterized data are passed into the algorithm.Based on the results listed in the table, YCrCb Channel-1 was selected classifier colorspace.  

Using similar method, I settled with the following parameter as a result of the trade-off. For example, by increasing histbin from 16 to 32, the accuracy increased from 97.3% to 98.4%. The final color parameters are as following:
```py
    spatial = 16
    histbin = 32
    colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 12
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 0 # Can be 0, 1, 2, or "ALL"
    hist_range = (0,1)
 ```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

 
I trained a linear SVM using selected HOG features, spatial binned color and color histogram. The classifier training code was implemented in the 6th cell under "Train classifier". I combined the three features in one function `extract features` and saved the results as `car_features` and `notcar_features`. 20% of the data were split as testing data using `sklearn.modelselection`. Before training, `StandardScaler()` was used to generate a scaler `Xscaler` to normalize and scale the training data. 

A linear support vector classification (linearSVC) was used to train the classifier.The model was saved in SVC class. 10 sets of test data was used to evaluate the prediction time, which will be discussed in the following session. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb channel-1 HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Two window sizes were used becaulse of the perspective issue. Vehicles near the camera are larger than the further ones. Therefore, a larger window (2.5) saved much simulation time. Different searching steps were used in different perspective as well.

```py
    y_start_stop = [390, 490] # Min and max in y to search in slide_window()
    y_start_stop2 = [390, 660] 

    windows1 = slide_window(image, x_start_stop=[300, None], y_start_stop=y_start_stop, 
                        xy_window=(72,72), xy_overlap=(0.85, 0.85))
    windows2 = slide_window(image, x_start_stop=[250, None], y_start_stop=y_start_stop2, 
                        xy_window=(128, 128), xy_overlap=(0.75, 0.75))
    windows = windows1+windows2
    
    hot_windows = search_windows(image, windows, svc, X_scaler, cspace=colorspace, spatial_size=(spatial, spatial),
                            hist_bins=histbin, hist_range=hist_range,orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel="ALL")

    window_img = draw_boxes(draw_image,hot_windows, color=(0, 0, 255), thick=6)
    
```


Also, since the vehicle could only appear on the road, only the bottom half of the image was scanned. Here is a view of the sliding windows:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output_th7_fine_step.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. 
From the positive detections I created a heatmap and then thresholded (2) that map to identify vehicle positions.
Results improved slightly, but still a few positive detections. Then I tried to use previous frame information to cross-verify the detection. 
```py
    def overlap(heatmap1, heatmap2):
        # Iterate through list of bboxes
        heatmap = heatmap1+heatmap2

        # Return updated heatmap
        return heatmap

```


I saved the heatmap of last frame and overlap the heatmap with the new one. Then I increased the threshold and output the overlap map. 
![alt text][image7]


To combine overlapping bounding boxes, `label()` function was used and `draw_labeled_bboxes` was implemented to draw the box of each detected vehicle on the map. 
```py
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
```


                                                                                                    
### Here are six frames, resulting bounding boxed and their corresponding heatmaps:

![alt text][image6]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


* False positive. Even I did overlap two frames and increased the threshold to eliminate some false positive, there were few existing in the video. 

* Computing speed. It took around 1 hour to process 1 minute video, which is unrealistic in the real time application. It might require more advanced search window algorithm and parallel computing techniques to make real-time processing possible. Reducing the "interested region" also speed up the computation. 

* Refine non-vehicle training data. The classifier still tends to recognize trees and barriers as vehicle. By including more common non-vehicle data set, the classifier might work better in the video. 

* Tree shadow. The classifier tends to recognize tree shadows as a vehicle. This will require additional shadow images as non-vehicle in the training set. 
