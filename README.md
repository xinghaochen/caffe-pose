
# Caffe-Pose

Modified Caffe that supports hand pose estimation.

## Features
- Data layers for ICVL, NYU, MSRA, Hands17 hand pose dataset
- PoseDistance layer to display mean joint error in training
- tools/output_pose to save predicted labels
- Baseline for hand pose estimation in ICVL, NYU, MSRA dataset

## Files have been changed (compared with [caffe@e963ef4](https://github.com/BVLC/caffe/tree/e963ef4eff86f07b67a23dfffc57f7e918959bd8))
- data_transformer.hpp/cpp (functions added and modified)
- io.hpp/cpp (functions added and modified)
- smooth_l1_loss_layer.hpp/cpp (added, from rbgirshick's [caffe-fast-rcnn](https://github.com/rbgirshick/caffe-fast-rcnn/tree/0dcd397b29507b8314e252e850518c5695efbb83))
- roi_pooling_layer.hpp/cpp (added, from rbgirshick's [caffe-fast-rcnn](https://github.com/rbgirshick/caffe-fast-rcnn/tree/0dcd397b29507b8314e252e850518c5695efbb83))
- pose_data_layer.hpp/cpp (added)
- pose_distance_layer.hpp/cpp (add)
- tools/output_pose (added)
- caffe.proto (parameters added and modified)

## Changelog

- Added tools/output_pose
- Added PoseDistanceLayer
- Added PoseDataLayer
- Merged ROIPoolingLayer from rbgirshick's [caffe-fast-rcnn](https://github.com/rbgirshick/caffe-fast-rcnn/tree/0dcd397b29507b8314e252e850518c5695efbb83)
- Merged SmoothL1LossLayer from rbgirshick's [caffe-fast-rcnn](https://github.com/rbgirshick/caffe-fast-rcnn/tree/0dcd397b29507b8314e252e850518c5695efbb83)
- Forked caffe from [caffe@e963ef4](https://github.com/BVLC/caffe/tree/e963ef4eff86f07b67a23dfffc57f7e918959bd8)

-------

# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by Berkeley AI Research ([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning Center (BVLC) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BAIR reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

## Custom distributions

 - [Intel Caffe](https://github.com/BVLC/caffe/tree/intel) (Optimized for CPU and support for multi-node), in particular Xeon processors (HSW, BDW, SKX, Xeon Phi).
- [OpenCL Caffe](https://github.com/BVLC/caffe/tree/opencl) e.g. for AMD or Intel devices.
- [Windows Caffe](https://github.com/BVLC/caffe/tree/windows)

## Community

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
