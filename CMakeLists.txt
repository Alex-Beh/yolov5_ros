cmake_minimum_required(VERSION 2.8)

project(yolov5_ros)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

find_package(OpenCV)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  std_msgs
  message_generation
  image_transport
)

catkin_package(
#  INCLUDE_DIRS include
   CATKIN_DEPENDS roscpp sensor_msgs std_msgs message_runtime image_transport cv_bridge
)

include_directories(
    include/${PROJECT_NAME}
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED include/${PROJECT_NAME}/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

add_executable(yolov5_engine src/calibrator.cpp src/yolov5.cpp)
target_link_libraries(yolov5_engine nvinfer cudart myplugins ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(yolov5_gesture src/video_inteference_yolov5.cpp)
target_link_libraries(yolov5_gesture nvinfer cudart myplugins ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_definitions(-O2 -pthread)