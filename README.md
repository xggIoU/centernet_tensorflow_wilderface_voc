# centernet_tensorflow_wilderface_voc
### 1. Introduction
* This is the unofficial  implementation of the "CenterNet:Objects as Points".In my experiment, it was not based on the DLA34, Hourglass and other networks in the original paper. I simply modified shufflenetv2_1.0x and yolov3, and kept their feature extraction part, then connected to centernet_detect_head, and did not use dcn convolution.
* This is just a simple attempt to the effect of the algorithm.I only have one 1080ti,**I did not use any data augmentation and any other tricks during training，so the model is not very good,still need more work to get good results**.If it helps you, please give me a star.You can read my Chinese notes.(1)<https://zhuanlan.zhihu.com/p/68383078>         (2)<https://zhuanlan.zhihu.com/p/76378871>
* Official implementation:<https://github.com/xingyizhou/CenterNet>
* CenterNet:Objects as Points:<https://arxiv.org/pdf/1904.07850.pdf>
* Shufflenetv2 is modified from:<https://github.com/timctho/shufflenet-v2-tensorflow>
* Shufflenetv2:<https://arxiv.org/abs/1807.11164>
* Yolov3 is is modified from:<https://github.com/wizyoung/YOLOv3_TensorFlow>
* Yolov3:<https://pjreddie.com/media/files/papers/YOLOv3.pdf>
* ExFuse:<https://arxiv.org/abs/1804.03821>
### 2. My experimental environment
* anaconda3、pycharm-community、python3.6、numpy1.14
* tensorflow1.13、slim
* cuda10.0、cudnn7.6
* opencv-python4.1
* gtx1080ti*1
### 3. datasets
* For single-target detection, trained on wilderface dataset with 12876 training images.
* For multi-target detection, trained on pascal-voc2012 dataset with 17125 training images.
### 4. Experimental
#### 4.1 Modified the heat map generation method to solve the problem that the loss cannot be optimized
![](https://github.com/xggIoU/centernet_tensorflow_wilderface_voc/blob/master/display_image/src_box.png)

pic1 src_bbox_gt

                                          
![](https://github.com/xggIoU/centernet_tensorflow_wilderface_voc/blob/master/display_image/heatmap_original.png)       ![](https://github.com/xggIoU/centernet_tensorflow_wilderface_voc/blob/master/display_image/heatmap_modified.png)

pic2 heatmap_original and heatmap_modified

![](https://github.com/xggIoU/centernet_tensorflow_wilderface_voc/blob/master/display_image/heatmap_original_box.png)  ![](https://github.com/xggIoU/centernet_tensorflow_wilderface_voc/blob/master/display_image/heatmap_modified_box.png)

pic3 heatmap_original_box and heatmap_modified_box

#### 4.2 Face detection(wilder face)
```
input_size:512x512
downsample_ratio:4.0
batch_size:14
global_steps:14800
epochs≈16
```
![](https://github.com/xggIoU/centernet_tensorflow_wilderface_voc/blob/master/display_image/face_detect.jpg)

pic4 shufflenetv2_face_result

#### 4.3 Multi-target detection(voc)
```
yolov3_centernet:
input_size:512x512
downsample_ratio:8.0
batch_size:8
global_steps:40000
epochs≈18

shufflenetv2_centernet:
input_size:512x512
downsample_ratio:4.0
batch_size:16
global_steps:40000
epochs≈37

shufflenetv2_seb_centernet:
input_size:512x512
downsample_ratio:4.0
batch_size:16
global_steps:40000
epochs≈37
```
##### 4.3.1 Network
![](https://github.com/xggIoU/centernet_tensorflow_wilderface_voc/blob/master/display_image/yolov3.jpg)

 pic5 yolov3_centernet_voc
                                                            
![](https://github.com/xggIoU/centernet_tensorflow_wilderface_voc/blob/master/display_image/shufflenet_net.png)

pic6 shufflenetv2_centernet_voc

![](https://github.com/xggIoU/centernet_tensorflow_wilderface_voc/blob/master/display_image/shufflenet_seb_net.png)

pic7 shufflenetv2_centernet_seb_voc

##### 4.3.2 result(on training set,not very good on the test set)
![](https://github.com/xggIoU/centernet_tensorflow_wilderface_voc/blob/master/display_image/voc_detect.jpg)

pic8 shufflenetv2_centernet_voc_result

#### 4.4 tensorboard loss curve
![](https://github.com/xggIoU/centernet_tensorflow_wilderface_voc/blob/master/display_image/yolov3_total_loss.svg)

pic9 yolov3_centernet_voc_total_loss

![](https://github.com/xggIoU/centernet_tensorflow_wilderface_voc/blob/master/display_image/shuffulenet_voc_total_loss.svg)

pic10 shufflenetv2_centernet_voc_total_loss

#### 4.5 inference time
```
environment：python3.6 gtx1080ti*1 intel-i7-8700k
model_name   			avg_time(ms)    input_size	 model_size(.pb)	
shufflenetv2_face_v1	        21.37	        512x512		 20.5MB
shufflenetv2_voc_v2		17.4		512x512		 24.9MB
yolo3_voc_v2		        25.53		512x512	         227.7MB

```
### 5. Run test demo(still need more work to get good results)
download ckpt file<https://pan.baidu.com/s/1OVtOyHdc6qgcvTn56s5m2w>code:qd35,and put them to ./shufflenetv2_face_V1/, ./shufflenetv2_seb_voc/, ./shufflenetv2_voc/,and ./yolo3_voc/,then run test_voc_on_images.py or test_face_on_images.py
### 6.Create tfrecords to train
* The function about how to create and parse tfrecords is under folder img2tfrecords_detection.
* You only need to modify the following variables：img_path, txt_path, tfrecords.
* Then run img2tfrecords_pad.py to create tfrecords and parse it by parse-tfrecords.py.
* For detailed implementation, please see the relevant code under folder img2tfrecords_detection.

