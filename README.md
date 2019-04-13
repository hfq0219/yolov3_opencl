# yolov3 的 opencl 实现 #

![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

# usage #
wget https://pjreddie.com/media/files/yolov3.weights

./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg

./darknet detector train cfg/coco.data cfg/yolov3.cfg darknet53.conv.74

./darknet detector valid cfg/coco.data cfg/yolov3.cfg yolov3.weights
