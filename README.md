# yolov5_ros

## Environments
We have tested this repository on both x86 Ubuntu 20.04 and Nvidia Xavier NX. 
- cuda: 11.2
- cudnn: 8.2.1
- tensorRT: 8.0.0

## How to use
You can set these model_name, nms_threshold, conf_thresh and batch_size in the following launch file
```
roslaunch yolov5_ros yolov5_ros.launch 
```

## Topics
    - Subscriber: /usb_cam/image_raw    [sensor_msgs/Image]
    - Publisher: /yolov5_video          [sensor_msgs/Image]