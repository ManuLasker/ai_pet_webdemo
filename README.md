# Deep Image Blending Web Demo
Ai Pet web demo using user feedback interface

This demo is based on DeepImageBlending paper, and it uses the grab cut OpenCV algorithm to mask regions of preferences from the source image. In this demo, the user is able to draw rectangles, white and black lines on the source image to select the objects for the blending process, and on the target image, it is necessary to draw a rectangle to know what are the position and the dimensions of the blending region.

In the next gif, there is a complete guide of how to use the app, reset the webpage every time you want to reuse the app to clean the cache.

![mask_helper](static/mask_grabcut.gif)