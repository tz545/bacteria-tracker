# Bacteria Tracker

A human-in-the-loop application that automates the detection and tracking of cells in microscope images and videos. 

## Table of Contents

[1. Project Description](#Project-Description)
[2. Installation](#Installation)
[3. Usage](#Usage)

### Project Description

The automated detection and tracking of cells in microscope images and videos is a difficult problem. Many software exist to tackle this problem, however, with different cell types often having very different sizes and shapes, ensuring accuracy in all cases is virtually impossible. 

Rather than attempting full automation of this process, this project is a human-in-the-loop approach designed for small-scale projects, particularly ones using more rarely-seen cell types that do not work with out-of-the-box cell detection software. Only a small amount of training data -- in our case only 10 labelled images -- is required to train an initial computer vision model, which can then assist in labelling additional images as it is corrected in the process of outputting data. The computer vision model is a shallower version of [UNet](https://arxiv.org/abs/1505.04597) (based on the assumption that there is minimal information that needs to be extracted across large regions of the images), which is easily trainable on free resources such as Google Colaboratory. 

### Installation

Python 3.8.13 is recommended for the install. Required libraries can be installed from within the project directory using 

```
pip install -r requirements.txt
```

Cell detection and manual correction are implemented on a Plotly Dash webapp. The script is src/dash_app/app.py, and can be run from anywhere. 

### Usage

The current Dash app demonstrates a particular application of this framework. In this project, the concentration of a fluorescent marker into one region of a bacterial cell -- "firing events" -- needs to be detected, and the duration and number of these firing events are desired outputs. The homepage on the Dash app gives detailed usage instructions for this project. 