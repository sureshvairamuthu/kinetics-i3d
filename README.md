# I3D models trained on Kinetics with RGB to Flow conversion

## Overview

This repository was forked from deepmind's repository (https://github.com/deepmind/kinetics-i3d) which contains trained models reported in the paper "[Quo Vadis,Action Recognition? A New Model and the Kinetics
Dataset](https://arxiv.org/abs/1705.07750)" by Joao Carreira and Andrew
Zisserman. The paper was posted on arXiv in May 2017, and will be published as a
CVPR 2017 conference paper.

### Setup

First follow the instructions for [installing
Sonnet](https://github.com/deepmind/sonnet).


Then, clone this repository using

`$ git clone https://github.com/deepmind/kinetics-i3d`

### Files

`evaluate_sample.py`

File reads numpy file for sample video stored under data folder
To run the example codef from deepmind 

`$ python evaluate_sample.py`

`i3d_load2videos.py` File allows to pass two types of files for two models for classifying.

RGB video for rgb model optical flow video for flow model
To run the file
`$ python i3d_load2videos.py --eval_type =joint --rgbvideo = [filename] --flowvideo= [filename]`

`i3d_flowconversion.py` File allows to pass one RGB video and converts it to opticalflow video if you choose joint or flow model for classification.

To run the file
`$ python i3d_flowconversion.py --eval_type =joint --rgbvideo = [filename]`


`opticalflow.py` File with optical flow conversion code. The path to the video file to be converted is added inside the code and not passed as an argument.

To run the file
`$ python opticalflow.py `





### Acknowledgments

Brian Zhang, Joao Carreira, Viorica Patraucean, Diego de Las Casas, Chloe
Hillier, and Andrew Zisserman helped to prepare this initial release. We would
also like to thank the teams behind the [Kinetics
dataset](https://arxiv.org/abs/1705.06950) and the original [Inception
paper](https://arxiv.org/abs/1409.4842) on which this architecture and code is
based.

