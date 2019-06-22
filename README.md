# centernet_tensorflow_wilderface_voc
### 1. Introduction
* This is the unofficial  implementation of the "CenterNet:Objects as Points".In my experiment, it was not based on the DLA34, Hourglass and other networks in the original paper. I simply modified shufflenetv2_1.0x and yolov3, and kept their feature extraction part, then connected to centernet_detect_head, and did not use dcn convolution.
* This is just a simple attempt to the effect of the algorithm.I only have one 1080ti,**I did not use any data augmentation and any other tricks during training，so the model is not very good,still need more work to get good results**.If it helps you, please give me a star.You can read my Chinese notes.<https://zhuanlan.zhihu.com/p/68383078>
* Official implementation:<https://github.com/xingyizhou/CenterNet>
* CenterNet:Objects as Points:<https://arxiv.org/pdf/1904.07850.pdf>
* Shufflenetv2 is modified from:<https://github.com/timctho/shufflenet-v2-tensorflow>
* Shufflenetv2:<https://arxiv.org/abs/1807.11164>
* Yolov3 is is modified from:<https://github.com/wizyoung/YOLOv3_TensorFlow>
* Yolov3:<https://pjreddie.com/media/files/papers/YOLOv3.pdf>
### 2. My experimental environment
* anaconda3、pycharm-community、python3.6、numpy1.14
* tensorflow1.12、slim
* cuda9.0、cudnn7.3
* opencv-python4.1
* gtx1080ti*1
### 3. datasets
* For single-target detection, trained on wilderface dataset with 12876 training images.
* For multi-target detection, trained on pascal-voc2012 dataset with 17125 training images.
### 4. Experimental result
#### 4.1 Face detection
```
input_size:512x512
downsample_ratio:4.0
batch_size:14
global_steps:14800
epochs≈16
train_time≈3.7 hours
```
##### 4.1.1 Network
![](https://github.com/xggIoU/centernet_tensorflow_wilderface_voc/blob/master/display_image/shufflenetv2_centernet.png)
##### 4.1.2 result
![](https://github.com/xggIoU/centernet_tensorflow_wilderface_voc/blob/master/display_image/face_detect.jpg)
#### 4.2 Multi-target detection
```
input_size:512x512
downsample_ratio:8.0
batch_size:8
global_steps:70000
epochs≈32
train_time≈9.7 hours
```
##### 4.2.1 Network
![](https://github.com/xggIoU/centernet_tensorflow_wilderface_voc/blob/master/display_image/yolov3_centernet.png)
##### 4.2.2 result(on training set,not very good on the test set)
![](https://github.com/xggIoU/centernet_tensorflow_wilderface_voc/blob/master/display_image/voc_detect.jpg)
#### 4.3 inference time
```
environment：python3.6 gtx1080ti*1 intel-i7-8700k
model_name   			avg_time(ms)    input_size	 model_size(.pb)	
shufflenet-face			21.37		512x512		 20.5MB
yolo3_centernet_voc		25.23		512x512		 230MB
```
### 5. Run test demo(still need more work to get good results)
download ckpt file<https://pan.baidu.com/s/1VrHv5U1wF1UP_r7JICbeZA>code:qqwx,and put them to ./shufflenet_face/ and ./yolo3_centernet_voc/,then run test_on_images.py
### 6.Create tfrecords to train
* The function about how to create and parse tfrecords is under folder img2tfrecords_detection.
* You only need to modify the following variables：img_path, txt_path, tfrecords.
* Then run img2tfrecords_pad.py to create tfrecords and parse it by parse-tfrecords.py.
* For detailed implementation, please see the relevant code under folder img2tfrecords_detection.

