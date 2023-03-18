# Computer Vision - Image stitching and panorama

Implemented background stitching and image panaroma as part of CSE 573:Computer Vision and Image Processing course at UB.

## Background stiching

Two images with similar background and different foreground are used to stitch the images to get one clean background. I have used homography between overlapping key features and RANSAC to optimize the result.

Run t1.py to for background stitching of 2 images read from the 'images' directory.
The resulting stitched image will be updated in the main directory. 

## Image Panorama

Mutiple images with some level of overlapping are stitched into one panoramic image.

Run t2.py to for panoramic image of multiple images read from the 'images' directory.
The resulting panoramic image will be updated in the main directory.
