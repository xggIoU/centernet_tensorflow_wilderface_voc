import tensorflow as tf
import numpy as np
import glob
import cv2

def py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5):
    """
    Pure Python NMS baseline.

    Arguments: boxes: shape of [-1, 4], the value of '-1' means that dont know the
                      exact number of boxes
               scores: shape of [-1,]
               max_boxes: representing the maximum of boxes to be selected by non_max_suppression
               iou_thresh: representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep[:max_boxes]

use_nms=True
input_img_size=512
down_ratio=4.0
center_score_thresh = 0.5
num_classes=20

backbone='yolov3' # yolov3,shufflenet,shufflenet_seb
class_names = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
imgfile_pattern = "demo_image_voc/*jpg"

if backbone=='shufflenet':
    model_path = "shufflenetv2_voc/shufflenetv2_voc.ckpt-41000"
    cls_tensor_name = 'shufflenet_centernet/detector/cls:0'
    size_tensor_name = 'shufflenet_centernet/detector/size:0'

elif backbone=='shufflenet_seb':
    model_path = "shufflenetv2_seb_voc/shufflenetv2_voc_SEB.ckpt-43000"
    cls_tensor_name = 'shufflenet_centernet/detector/cls:0'
    size_tensor_name = 'shufflenet_centernet/detector/size:0'

elif backbone=='yolov3':
    down_ratio = 8.0
    model_path = "yolov3_voc/yolov3_voc.ckpt-40000"
    cls_tensor_name = 'yolo3_centernet/detector/Conv_1/Sigmoid:0'
    size_tensor_name = 'yolo3_centernet/detector/Conv_3/BiasAdd:0'

else:
    print('backbone only supports yolov3 or shufflenet...')

#coco class
# class_names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

imgfile=glob.glob(imgfile_pattern)

sess = tf.Session()
saver = tf.train.import_meta_graph(model_path + ".meta")
saver.restore(sess, model_path)

#get input and output tensors by tensor name
input_tensor = sess.graph.get_tensor_by_name('inputs:0')
input_training = sess.graph.get_tensor_by_name('is_training:0')
output_cls=sess.graph.get_tensor_by_name(cls_tensor_name)
output_size = sess.graph.get_tensor_by_name(size_tensor_name)

#filter center points by max pool and center score
output_peak=tf.layers.max_pooling2d(output_cls,5,1,padding='same')
peak_mask=tf.cast(tf.equal(output_cls,output_peak),tf.float32)
score_mask=tf.cast(tf.greater_equal(output_cls,center_score_thresh),tf.float32)
obj_mask=peak_mask*score_mask
output_cls=output_cls*obj_mask

for imagename in imgfile:
    src=cv2.imread(imagename)
    height,width=src.shape[0:2]
    max_size = max(height, width)
    scale=1.0
    if max_size <= input_img_size:
        top = (input_img_size - height) // 2
        bottom = input_img_size - top - height
        left = (input_img_size - width) // 2
        right = input_img_size - left - width
    else:  # max_size>input_img_size
        if height >= width:
            scale = input_img_size / height
            height = input_img_size
            width = int(width * scale)
            top = 0
            bottom = 0
            left = (input_img_size - width) // 2
            right = input_img_size - left - width
        else:
            scale = input_img_size / width
            width = input_img_size
            height = int(height * scale)
            top = (input_img_size - height) // 2
            bottom = input_img_size - top - height
            left = 0
            right = 0
    img_resize = cv2.resize(src, (width, height))
    img_resize = cv2.copyMakeBorder(img_resize, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    src_resize = img_resize.copy()
    img_resize=img_resize.astype(np.float32)/255.0-0.5
    input_img=np.expand_dims(img_resize,axis=0)
    cls,size=sess.run([output_cls,output_size],feed_dict={input_tensor:input_img,input_training:False})

    for i in range(num_classes):

        cls_heatmap=cls[0,:,:,i]#[128,128]
        cords = np.argwhere(cls_heatmap>0.0)

        boxes=[]
        scores=[]
        for cord in cords:

            w=size[0,cord[0],cord[1],0]* down_ratio
            h = size[0,cord[0],cord[1],1]* down_ratio

            x1 = (cord[1] +0.5) * down_ratio-w/2
            y1=(cord[0]+0.5)*down_ratio-h/2
            x2=(cord[1]+0.5)*down_ratio+w/2
            y2=(cord[0]+0.5)*down_ratio+h/2
            score=cls_heatmap[cord[0],cord[1]]

            #remove padding and map to original scale
            if top == 0 and bottom == 0:
                x1_src = int((x1 - left) / scale)
                y1_src = int(y1 / scale)
                x2_src = int((x2 - left) / scale)
                y2_src = int(y2 / scale)
            elif left == 0 and right == 0:
                x1_src = int(x1 / scale)
                y1_src = int((y1 - top) / scale)
                x2_src = int(x2 / scale)
                y2_src = int((y2 - top) / scale)
            else:
                x1_src = int(x1 - left)
                y1_src = int(y1 - top)
                x2_src = int(x2 - left)
                y2_src = int(y2 - top)
            if use_nms:
                boxes.append([x1_src, y1_src,x2_src, y2_src])
                scores.append(score)
            else:
                txt=class_names[i]+":"+str(round(score,2))
                center_x_src = (x1_src + x2_src) // 2
                center_y_src = (y1_src + y2_src) // 2
                cv2.putText(src, txt, (x1_src, y1_src - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                cv2.circle(src, (center_x_src, center_y_src), 2, (0, 0, 255), 2)
                cv2.rectangle(src, (x1_src, y1_src), (x2_src, y2_src), (0, 255, 0), 2)

        #nms
        if use_nms and boxes!=[]:
            boxes=np.asarray(boxes)
            scores=np.asarray(scores)
            inds = py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5)
            for ind in inds:
                x1_src = boxes[ind][0]
                y1_src = boxes[ind][1]
                x2_src = boxes[ind][2]
                y2_src = boxes[ind][3]

                txt=class_names[i]+":"+str(round(scores[ind],2))
                center_x_src = (x1_src + x2_src) // 2
                center_y_src = (y1_src + y2_src) // 2
                cv2.putText(src, txt, (x1_src, y1_src - 2),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                cv2.circle(src, (center_x_src, center_y_src), 2, (0, 0, 255), 2)
                cv2.rectangle(src, (x1_src, y1_src), (x2_src, y2_src), (0, 255, 0), 2)

    cv2.imshow('src_box', src)
    cv2.waitKey()
