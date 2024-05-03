
# Evaluation of Object Detection and Object Tracking Models for Autonomous Vehicles

## PointPillars.md
Environment setup instructions for the PointPillars object detection framework.

## PTTR.md
Environment setup instructions for the PTTR object tracking framework.

## carla_to_kitti.py
This script converts spatial data from the CARLA simulator into KITTI-format files for evaluation. 

It processes transformation files and bounding box data into corresponding matrices, and then converts them into a format suitable for object detection and tracking tasks. The output includes KITTI-format files and calibration data, which are saved to output /kitti_output.

Running this script assumes you have a /carla data folder structured with /bbox, /car, and /infra subdirectories.

Make sure to update FRAME_COUNT if your data has more frames than the example data.

Here is some example [Carla data](https://drive.google.com/drive/folders/1nac6HKlk4IUrvDkLjMpMASxbKAM2QwV3?usp=sharing).
    
NOTE: This script does not assign object labels in the KITTI output. These are 'DontCare' by default.


## mp4_to_img.py

This script converts a .mp4 into a series of images.

In this case the images generated are based on /carla/infra/i_timestamp timestamps.

