# Learned Scalable Video Coding for Humans and Machines

This repository presents a PyTorch implementation for the following paper:

H. Hadizadeh and I. V. Bajić, “Learned scalable video coding for humans and machines,” arXiv preprint arXiv:2307.08978, Jul. 2023: https://arxiv.org/abs/2307.08978


![The block diagram of the proposed learned scalable video compression system. The base layer has two outputs: the base frame and the latent features by which the computer vision task is performed. The
enhancement layer has only one output, which is the reconstructed frame.](flowchart.png)

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)

## Installation

First, clone this repository into a local folder on your machine. Then go to the main folder of the repository, and install YOLOv5 as described in https://github.com/ultralytics/yolov5. This installs all the required packages to run this project including torch 2.4.1, numpy 1.24.4, and torchvision 0.19.1.

After that, get the checkpoints from this google drive, and put them in ./checkpoints: https://drive.google.com/drive/folders/1F9TnUgYBxja9DnRT6jgGdZb2EMEsNnvr?usp=sharing

## Usage
To test the base layer of the proposed system, begin by converting your input video into RGB frames and saving them in a folder, such as ./input. You can use ffmpeg for this conversion. For example, to convert a YUV420 video into PNG frames, use the following command:
```bash
ffmpeg -pix_fmt yuv420p -s 1920x1080 -i BasketballDrive_1920x1080_50.yuv -f image2 ./input/im_%03d.png
```

In the command above, "im_" will serve as the prefix for each video frame's filename. For instance, the 4th frame will be named im_004.png.

You can then use the following command to encode the video in the base layer and save the decoded base frames in an output folder, such as ./out/:
```bash
python test_base.py --inp_path ./input/ --out_path ./out/ --prefix im_ --checkpoint_number 1 --no_frames 100 --gop 32
```

In the above example, we assume the input video consists of 100 frames, and the GOP (Group of Pictures) or intra period size is set to 32. The checkpoint number is an integer ranging from 1 to 4. If you set the checkpoint number to 1, the model with the lowest rate will be used. Conversely, if you set it to 4, the model with the highest rate will be used.

To test the enhancement layer of the proposed system, use the following command instead:
```bash
python test_enh.py --inp_path ./input/ --out_path ./out/ --prefix im_ --checkpoint_number 1 --no_frames 100 --gop 32
```

If you have any questions, please contact the main author at hadi[dot]tutorials at gmail dot com.
