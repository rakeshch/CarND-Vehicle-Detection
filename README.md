**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector.
* For those first two steps normalize extracted features and randomize a selection for training and testing.
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Features extraction

#### 1. Extract features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images. This can be found in the code cell titled 'Data Exploration'. Some examples of the `car` and `non-car` classes can be seen under title "Visualizing some images in dataset" in my project file and can also be seen below:

![alt text][image1]

Once I explored the data that I will use in this project, the next step is to extract features from images. In this project, I used a combination of Histogram of Oriented Gradients (HOG) features, Histogram of color features and Spatial binning of color features. I started with using only HOG features for training my classifier but combining with other features did result in better results.

Histogram of Orieneted Gradients (HOG): 
The code for this step can be found under title "Histogram of Oriented Gradients (HOG)" in my project file. 

I have used scikit-image package that has a built in function to extract Histogram of Oriented Gradient features. The documentation for this function can be found [here](http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=feature%20hog#skimage.feature.hog) and a brief explanation of the algorithm and tutorial can be found [here](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html).Steps in this algorithm can be summarised as:

1. (optional) global image normalisation  
2. computing the gradient image in x and y
3. computing gradient histograms
4. normalising across blocks
5. flattening into a feature vector

I have explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like in different color spaces. The code and examples for this visualization can be seen under title 'Visualizing HOG on an image from dataset'. Below are the examples using different color spaces and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:

![alt text][image2]

Histogram of Color:
The code for this step can be found under title "Histogram of Color" in my project file. 

Histogram of color in an image is useful in differentiating objects but this could wrongly classfy a non-car image as a car image when we solely rely on distribution of color values. Due to this reason we combine these features with other features to help differentiate the objects in an image. One example of the `car` and `non-car` class can be seen under title "Visualizing Hisogram of color on an image from dataset" in my project file and can also be seen below: 

![alt text][image2]

Spatial Binning of color:
The code for this step can be found under title "Spatial Binning of color" in my project file. 

As seen in Histogram of color example above, template matching is not a particularly robust method for finding vehicles unless you know exactly what your target object looks like. However, raw pixel values are still quite useful to include in your feature vector in searching for cars.

While it could be cumbersome to include three color channels of a full resolution image, you can perform spatial binning on an image and still retain enough information to help in finding vehicles. In order to reduce the number of elements in feature vector I have used the size (16,16) and I could still retain enough information from the image. One example of the `car` and `non-car` class can be seen under title "Visualize spatial binning on an image from dataset" in my project file and can also be seen below: 

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

Parameter tuning was the most time taking and fun part of this project. I tried various combinations of parameters with RGB, HSV, YCrCb, HLS, YUV, LUV color spaces and HOG parameters with `orientations` values of 6,7,8,9,11,12,16 and `pixels_per_cell` values of (8, 8),(16, 16) and `cells_per_block` values of (2,2), (3,3), (4,4). I settled on my final choice of HOG parameters based upon the performance of the SVM classifier. I considered not only the accuracy with which the classifier made predictions on the test dataset, but also the speed at which the classifier is able to make predictions.

The final parameters chosen were: LUV colorspace, 11 orientations, 16 pixels per cell, 2 cells per block, and ALL channels of the colorspace.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the section titled "Preparing data for classifier" I extracted all the features from dataset by combining HOG features, Histogram of color features, Spatial binning of color which resulted in a feature vector of length 2052. I then defined labels for features, shuffled the data, split the data into training and test data and normalised features.

Once I have the data for the calssifier, in the section titled "Train a Classifier" I trained a linear SVM with C=0.01 and using all the features extracted as explained in above steps and was able to achieve a test accuracy of 98.67%. With default parameters for linear SVM my test accuracy was 98.42%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

