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

#### Usage example

The current Dash app demonstrates a particular application of this framework. In this project, the concentration of a fluorescent marker into one region of a bacterial cell -- "firing events" -- needs to be detected, and the duration and number of these firing events are desired outputs.  

The Dash app provides 

1. A graphical interface for selecting, deleting, and modifying cells, either completely from scratch, or by correcting the outputs of a pre-trained model:  
<img src="https://github.com/tz545/bacteria-tracker/blob/master/src/dash_app/assets/lasso_select.png" width="250">


2. Automated cell tracking across frames of a video, with the ability to incorporate corrections provided by the user:    
![cell tracking](src/dash_app/assets/cell_tracking.png?raw=true)

3. Automated detection of firing events based on a threshold provided by the user. This is accompanied by a visualization which aids the user in adjusting the threshold.  
![firing detection adjustable threshold](src/dash_app/assets/threshold_adjust.png?raw=true)


#### Adapting to other projects

This human-in-the-loop approach requires the following two generalized steps, which are independent of project specifics:  

1. Data labelling/correction: manual labelling of initial set of training data, or the correction of labelled data generated by an imperfect model.  
2. Model training/re-training: training or re-training the model with correctly labelled images for increasingly better performance.  

The **Boundary detection** page in the Dash app (`boundary_detection.py`) contains all the elements necessary for performing the initial data labelling and further corrections. In particular, the callbacks `update_cells` and `update_figure` enable the user to select, delete and modify cell selections. By default these callbacks support the tracking of cells across consecutive frames (as in the usage example), which can be easily removed. The callback `download_labels` saves the user-corrected labels in a format ready to feed into the model training step. There is also built-in support for selecting a pre-trained model for cell detection, or applying simple threshold-based segmentation (e.g. in the absence of a trained model).  

The remaining callbacks in `boundary_detection.py`, and the additional **Firing detection** (`firing_detection.py`) support specific functionalities to the usage example, and can be removed or ignored.  

The pipeline for model training/re-training is contained in the `unet_wandb.ipynb`. This sample notebook uses the [Weights & Biases](https://wandb.ai/site) platform for hyperparameter optimization and model training.  
