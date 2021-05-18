# Image Processing System for Traffic Lights using Deep Learning

Authors:  Viviana Gherasim, Gornic Diana, Getejanc Marin,
            3rd Year Students of Computers and Information Technology, Politehnica University of Timisoara, 2021 

General information about the system
 
·      The system consists of an image classification algorithm, based on a deep learning algorithm running on a Raspberry Pi 4 with a video camera attached.
·      Hardware Components:
1.	 Raspberry Pi 4 with the following specification:
- 1.5GHz Broadcom BCM2711, quad-core Cortex-A72 (ARM v8) 64-bit
- 4GB LPDDR4-2400 SDRAM
- 2.4GHz i 5.0GHz IEEE 802.11b/g/n/ac wireless LAN
- Bluetooth 5.0, BLE
- Gigabit Ethernet
- 2x USB 2.0, 2x USB 3.0 ports
- Standard 40-pin GPIO header
- 2-lane MIPI DSI display port
- 2-lane MIPI CSI video camera connector
- 4-pole audio/video out
- H.265 (4Kp60 decode), H.264 (1080p60 decode, 1080p30 encode), OpenGL ES, 3.0 graphics
- 64gb SanDisk Ultra microSDXC Card (with RaspbianOS installed on it)
2.	5Mpix Video Camera capable of 1080p@30fps, 640x480@60/90fps
 
 
·      Software Component:
1.	OpenCV component:
-Popular library used for image processing (Thresholding, Cascading, HOG, etc)
2.	Convolutional Neural Network model:
-Deep neural network architecture specific for computer vision solutions (image classification, object detection, object tracking, semantic segmentation, instance segmentation, etc)
 
	The software components are written in Python 3, the libraries being used are:
OS - Library used for OS operations
Shutil - Library used for easy moving of files
Numpy - Library used for tensor operations
OpenCV - Library used for classic computer vision operations
 Tensorflow - Library used for deep learning

Use Case
 
This project is more of an example of a proof-of-concept for image processing and computer vision, which could be adapted for real world use (for example in robotics or autonomous driving). The main components and concepts still apply to real world problems, the main difference being in the neural network model and the need of additional components.


For more information read the following document: 

https://docs.google.com/document/d/1gppiM_NYEqbhNMjnled6vkdeFPl8RXiGxe36y9DEL4Y/edit?usp=sharing
