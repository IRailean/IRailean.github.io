---
published: true
title: Dense optical flow with Python using OpenCV
layout: single
author_profile: true
read_time: true
categories: [projects]
header :
    teaser: "/assets/images/hunter-harritt-Ype9sdOPdYc-unsplash.jpg"
comments : true
toc: true
toc_sticky: true
sidebar:
    nav: sidebar-sample
---

# Dense optical flow with Python using OpenCV.

Optical flow can arise from the relative motion of objects and the viewer. It has a huge variety of applications. For example segmentation, or object detection and tracking. Today's goal is to implement the Gunnar Farneback algorithm in Python to determine dense optical flow in a video. As an example, we`ll take this video of moving cars.

<iframe src='https://gfycat.com/ifr/LoathsomeFelineAmericanshorthair' frameborder='0' scrolling='no' allowfullscreen width='640' height='404'></iframe>

You can check out [this](https://www.youtube.com/watch?v=a-v5_8VGV0A&t=61m30s) video lecture, that perfectly explains how does Farneback algorithm work. 
Also, you can find source code that will be shown below on [Github](https://github.com/IRailean/Dense-Optical-Flow).
Enough introduction, let`s get to the code.

## **Preparing video and reading first frame**

OpenCV and numpy will be used, so import them.

    import cv2
    import numpy as np

Then we have to indicate the video source. Set it as a parameter to [cv2.VideoCapture()](https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-videocapture), or simply type 0 to get input from your webcam. (If you`re using an external webcam type 1). Then read first frame using [vc.read()](https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-read). Next, resize image to 600x600 (you can choose size on your own) and convert color scheme to grayscale using [cv2.cvtColor()](https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor).

    # Get a VideoCapture object from video and store it in vÐü
    vc = cv2.VideoCapture(“Car.mp4”)
    # Read first frame
    _, first_frame = vc.read()
    # Scale and resize image
    resize_dim = 600
    max_dim = max(first_frame.shape)
    scale = resize_dim/max_dim
    first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
    # Convert to gray scale 
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

Also, we will need a mask to combine calculated optical flow and original video. It will be used in HSV color format. Mask will be used in the following way: hue will indicate the direction, where the car is moving, value will indicate how fast it is going (See gif above). As we do not need saturation component, we will simply set it to its maximum value.

    # Create mask
    mask = np.zeros_like(first_frame)
    # Set image saturation to maximum value as we do not need it
    mask[…, 1] = 255

## **Determining dense optical flow**

So, we have set up the video stream, read the first frame, and even defined the mask. Now we are ready to read the video input frame by frame. Let`s write a loop for this. As we want to process the whole video we will read frames until [vc.isOpened()](https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-isopened) returns true, i.e. while we have not reached the end of the video. Similarly to the first frame, we will resize each frame and convert it to grayscale.

    while(vc.isOpened()):
     # Read a frame from video
     _, frame = vc.read()
     
     # Convert new frame format`s to gray scale and resize gray frame obtained
     gray = cv2.resize(frame, None, fx=scale, fy=scale)
     gray = cv2.resize(gray, None, fx=scale, fy=scale)

Here is really the core part of this program. Optical flow will be calculated using [cv2.calcOpticalFlowFarneback()](https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback) . prev_gray and gray are the previous and the current frame. After that, you will have to set some other important parameters. Why did we use exactly these parameters? W̶e̶ ̶d̶o̶ ̶n̶o̶t̶ ̶a̶c̶t̶u̶a̶l̶l̶y̶ ̶k̶n̶o̶w̶ Because of optimization reasons. You can play here and choose different parameters and see how it will change the results.
Then, compute the magnitude and angle of the 2D vectors (flow vectors) to indicate direction and speed of a moving car via [cv2.cartToPolar()](https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#carttopolar). Set image hue and value according to magnitude and angle. Finally, convert it back to BGR color scheme.

    # Calculate dense optical flow by Farneback method
     flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
     # Compute the magnitude and angle of the 2D vectors
     magnitude, angle = cv2.cartToPolar(flow[…, 0], flow[…, 1])
     # Set image hue according to the optical flow direction
     mask[…, 0] = angle * 180 / np.pi / 2
     # Set image value according to the optical flow magnitude (normalized)
     mask[…, 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
     # Convert HSV to RGB (BGR) color representation
     rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

Let`s see how calculated dense optical flow looks like for our video

<iframe src='https://gfycat.com/ifr/FirsthandSameGarpike' frameborder='0' scrolling='no' allowfullscreen width='640' height='407'></iframe>

Great! Now, we have to combine the mask and original video frame and show the resulting frame. Do not forget to update the previous frame.
Also, we want to read frames each 10 ms, that is why [cv2.waitKey() ](https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=waitkey#waitkey)function is added.

    # Resize frame size to match dimensions    
    frame = cv2.resize(frame, None, fx=scale, fy=scale)
    # Open a new window and displays the output frame
     dense_flow = cv2.addWeighted(frame, 1,rgb, 2, 0)
     cv2.imshow("Dense optical flow", dense_flow)
     # Update previous frame
     prev_gray = gray
     # Frame are read by intervals of 10 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
     if cv2.waitKey(10) & 0xFF == ord('q'):
     break

Finally, do not forget to do the cleaning

    vc.release()

    cv2.destroyAllWindows()

## Conclusion

We have seen how to calculate dense optical flow for a video in just 60 lines of Python code. You can also play around with parameters for cv2.calcOpticalFlowFarneback to see how it will change the obtained results.

## References

* [https://www.youtube.com/watch?v=a-v5_8VGV0A&list=PLjMXczUzEYcHvw5YYSU92WrY8IwhTuq7p&index=8](https://www.youtube.com/watch?v=a-v5_8VGV0A&list=PLjMXczUzEYcHvw5YYSU92WrY8IwhTuq7p&index=8)

* [https://www.researchgate.net/publication/225138825_Two-Frame_Motion_Estimation_Based_on_Polynomial_Expansion](https://www.researchgate.net/publication/225138825_Two-Frame_Motion_Estimation_Based_on_Polynomial_Expansion)

* [https://nanonets.com/blog/optical-flow/](https://nanonets.com/blog/optical-flow/)

## Code available on Github:

* [https://github.com/IRailean/Dense-Optical-Flow](https://github.com/IRailean/Dense-Optical-Flow)

Let me know your feedback. If you like it, please recommend and share it. Thank you.
