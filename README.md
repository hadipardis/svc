# Project Name

A Python implementation for "Learned Scalable Video Coding For Humans and Machines".
Written by Hadi Hadizadeh, 2024.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)

## Installation

First, clone this repository into a local folder on your machine. Then go to the folder, and install YOLOv5 as described in https://github.com/ultralytics/yolov5. This installs all the required packages to run this project.

## Usage
To test the base layer of the proposed system, first convert your input video into RGB frames and put them in a folder such as .\input. For this purpose, you can use ffmpeg. For example, to convert a YUV420 video into RGB frames using ffmpeg, you can use this command:
```bash
ffmpeg -pix_fmt yuv420p -s 1920x1080 -i BasketballDrive_1920x1080_50.yuv -f image2 ./input/im%03d.png

