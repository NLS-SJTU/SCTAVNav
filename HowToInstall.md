# How To Install BNavi+
## Introduction
This document describes how to install all necessary components of this autonomous vision navigation system. The following chapters are installations of different modules. The system is tested in Ubuntu 16.04 and Ubuntu 18.04.
**Before the installation, please ensure that ROS-desktop-full and Anaconda are installed.**

## Elevation Mapping
### Introduction
This [module](git@github.com:NLS-SJTU/elevation_mapping.git) (or the original [version](https://github.com/ANYbotics/elevation_mapping.git)) builds elevation map for local planning.

### Installation
Just follow the instructions in the readme.md of that repository. Dependencies of this module can be install by apt install ros-*-*. We recommand you to put this repository in a isolated catkin workspace with Semantic Elevation Map Planner.

    cd catkin_ws_iso/src
    git clone git@github.com:NLS-SJTU/elevation_mapping.git
    cd ..
    catkin_make_isolated

### Run
In Gazebo:

    roslaunch elevation_mapping_demos mytest_gazebo.launch

In real robot (start your sensors first):

    roslaunch elevation_mapping_demos mytest_zed.launch  (with only zed)
    roslaunch elevation_mapping_demos mytest_kinectzed.launch   (with kinect)

## Semantic Elevation Map Planner
### Introduction
This [module](git@github.com:NLS-SJTU/sele_path_planner.git) plans local motion with elevation map.

### Installation
Follow the steps. 

    cd catkin_ws_iso/src
    git clone git@github.com:NLS-SJTU/sele_path_planner.git
    cd ..
    catkin_make_isolated

### Run
In Gazebo:

    roslaunch sele_path_planner testgazebo.launch

In real robot:

    roslaunch sele_path_planner testzed.launch

## HF-Net
### Introduction
This module ([page](https://projects.asl.ethz.ch/datasets/doku.php?id=cvpr2019hfnet) and [github](https://github.com/ethz-asl/hfnet.git)) extracts global and local features of an image for localization.

### Installation
Create an anaconda environment, download codes, install the requirement depedencies and download the trained weights of this model. Then, change *EXPER_PATH* in [global_cfg.py](./src/utils/global_cfg.py) to your path with the trained weights.

## SCTAVNav
### Introduction
This [module](https://github.com/NLS-SJTU/SCTAVNav.git) have the mapping, localization and global planning functions. It can be put to any folder. There are several modules, which are localizer, planner and mapper.

### Installation
Localizer needs a conda environment to run in opencv-4.5. The other modules can run with default python environment. The dependencies are in requirement.txt. In conda, PyKDL can be search by key word orocos-kdl. Just pip install them.

Open BrainNaviUI/src/hfnet_main.py and set EXPER_PATH to the folder of your saved model of hfnet.

### Run
Go into folder BrainNaviUI/src/ and follow the steps:

    (autonomous navigation)
    (new terminal) python main_plan.py
    (new terminal) python main_core.py
    (new terminal) python main_ros.py
    (new terminal and activate corresponding conda env) python main_loc.py
    (new terminal and activate corresponding conda env) python main_hfnet.py
    
    (mapping)
    (new terminal and activate corresponding conda env) python main_hfnet.py
    (new terminal) python main_mapping.py

### Usage
* Click button *locsrc* and set it to phone. 
* Click button *loc/nav/sup* to set model to localization or navigation. 
* If you are in navigation mode, double click a node in the map to set destination. In this version, you should choose the solid nodes of intersections as destination, or some bugs will appear. 
* Click *start/pause* button to start.

## Author
Wuyang Xue
