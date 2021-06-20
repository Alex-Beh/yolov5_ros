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

## How to generate the engine file
Engines are specific to the exact hardware and software they were built on. Below are the steps to build the engine file with your CPU/GPU.
1. Remember to checkout the v5.0 branch in ultralytics/yolov5
2. Train your custom model with pretrained weight
    python train.py --img 640 --batch 32 --epochs 500 --data custom_model.yaml --weights yolov5l.pt
2. Generate .wts from pytorch with .pt usging the [gen_wts.py](script/gen_wts.py) script
3. Remember to update the Number of classes defined in yololayer.h
4. Build the repo and generate the engine file with the following command:
    rosrun yolov5_ros yolov5_engine -s (path-to- .wts file) custom_model.engine l

## Credits
1. [yolov5](https://github.com/ultralytics/yolov5)
2. [tensorRT](https://github.com/wang-xinyu/tensorrtx) 