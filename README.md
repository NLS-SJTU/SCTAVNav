# Autonomous Vision Navigation using Semi-consistent TopoMetric Maps
## Introduction
This is the open source code for paper "Autonomous Vision Navigation using Semi-consistent TopoMetric Maps". The system is tested in Ubuntu 16.04 and Ubuntu 18.04.

## Installation
See [HowToInstall.md](./HowToInstall.md).

## Run
Details can be found in HowToInstall.md. Before running, please check the topic names of ROS nodes.
### Mapping
* launch Gazebo (simulation) or sensors (real world).
* start hfnet.
* start topomapping in SCTAVNav.

### Autonomous Navigation
* launch Gazebo (simulation) or sensors (real world).
* launch elevation_map and sele_path_planner.
* start autonomous navigation in SCTAVNav.

## Author
Wuyang Xue