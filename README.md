# centernet_tensorflow_wilderface_voc
### 1.Introduction
This is the unofficial  implementation of the "CenterNet:Objects as Points".In my experiment, it was not based on the DLA34, Hourglass and other networks in the original paper. I simply modified shufflenetv2_1.0x and yolov3, and kept their feature extraction part, then connected to centernet_detect_head, and did not use dcn convolution.
<br>This is just a simple attempt to the effect of the algorithm.
<br>Official implementation:<https://github.com/xingyizhou/CenterNet>Corresponding paper:<https://arxiv.org/pdf/1904.07850.pdf>
<br>Shufflenetv2 is modified from:<https://github.com/timctho/shufflenet-v2-tensorflow>
<br>Yolov3 is is modified from:<https://github.com/wizyoung/YOLOv3_TensorFlow>

### 2.My experimental environment
* anaconda3、pycharm-community、python3.6、numpy1.14
* tensorflow1.12、slim
* cuda9.0、cudnn7.3
* opencv-python4.1
* gtx1080ti*1
### 3.datasets
<br>for single-target detection, trained on wilderface dataset with 12876 training images.
<br>for multi-target detection, trained on pascal-voc2012 dataset with 17125 training images.
### 4.Experimental result
#### 4.1 Face detection
```
input_size:512x512
batch_size:14
global_steps:14800
epochs≈16
train_time≈3.7 hours
```
##### 4.1.1 Network

