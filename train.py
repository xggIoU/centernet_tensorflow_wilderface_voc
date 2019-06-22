# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
import cv2
import cfg
from shufflenetv2_centernet import ShuffleNetV2_centernet
from yolov3_centernet import yolov3_centernet
from create_label import CreatGroundTruth

def parse_color_data(example_proto):
    
    features = {"img_raw": tf.FixedLenFeature([], tf.string),
               "label": tf.FixedLenFeature([],tf.string),
                "width": tf.FixedLenFeature([], tf.int64),
                "height": tf.FixedLenFeature([], tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    img = parsed_features["img_raw"]
    img = tf.decode_raw(img, tf.uint8)
    width=parsed_features["width"]
    height=parsed_features["height"]
    img=tf.reshape(img,[height,width,3])
    img = tf.cast(img,tf.float32) * (1./255.) - 0.5
    label = parsed_features["label"]
    label=tf.decode_raw(label,tf.float32)
    
    return img,label
 
def erase_invalid_val(sequence):
    label=[]
    h,w=sequence.shape
    mask=(sequence!=-1.0)
    for i in range(h):
        seq_new=sequence[i][mask[i]]
        label.append(list(seq_new))
    return label


filenames = [cfg.tfrecords_path]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.shuffle(buffer_size=1500)
dataset = dataset.map(parse_color_data)
# dataset = dataset.batch(128)
val1=tf.constant(-0.5,tf.float32)
val2 = tf.constant(-1, tf.float32)
dataset = dataset.padded_batch(cfg.batch_size, padded_shapes=([None, None, 3], [None]), padding_values=(val1, val2))
dataset = dataset.repeat(cfg.epochs)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

train_start_time = cv2.getTickCount()

if cfg.num_classes==1:
    model=ShuffleNetV2_centernet(model_scale=1.0, shuffle_group=2)
else:
    model=yolov3_centernet()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter("summary-face", sess.graph)
saver=tf.train.Saver(max_to_keep=20)

batch_num = 1
if 0:#reload model
    model_file = tf.train.latest_checkpoint('shufflenet-face/')
    # model_file = "yolo3_centernet_voc/yolo3_centernet_voc.ckpt-3200"
    saver.restore(sess, model_file)
    print("reload ckpt from "+model_file)
try:
    while True:
        batch_start_time=cv2.getTickCount()
        img_batch, label_batch = sess.run(next_element)
        label_batch = erase_invalid_val(label_batch)
        # print(label_batch)
        center_gt_batch, offset_gt_batch, size_gt_batch, mask_gt_batch = CreatGroundTruth(label_batch)
        feed = {model.inputs: img_batch,
                model.is_training:True,
                model.center_gt:center_gt_batch,
                model.offset_gt:offset_gt_batch,
                model.size_gt:size_gt_batch,
                model.mask_gt:mask_gt_batch
               }
        fetches = [model.pred_center,
                   model.pred_offset,
                   model.pred_size,
                   model.cls_loss,
                   model.offset_loss,
                   model.size_loss,
                   #model.regular_loss,
                   model.total_loss,
                   model.global_step,
                   model.lr,
                   model.merged_summay,
                   model.train_op,
                   ]
        pred_center,pred_offset,pred_size,cls_loss,offset_loss,size_loss,total_loss,global_step, lr, summary, _ = sess.run(fetches, feed)
        train_writer.add_summary(summary, global_step)
        time_elapsed = (cv2.getTickCount()-batch_start_time)/cv2.getTickFrequency()
        if batch_num%50==0:
            # print(pred_center[0, :, :, :])
            print(np.round(np.max(pred_center[0, :, :, :]),5))
            print(np.round(np.min(pred_center[0, :, :, :]),5))
            # print(pred_offset[0,:, :, 1])
            # print(pred_size[0,:, :, 1])
        #保存
        if batch_num%200==0:
            saver.save(sess,"shufflenet-face/shufflenet-face.ckpt",global_step=global_step)
        if batch_num % 10 == 0:
            print("-------Training {0}th batch-------".format(batch_num))
            print("global_step:{0} total_loss:{1:0.3f} cls_loss:{2:0.3f} offset_loss:{3:0.3f} size_loss:{4:0.3f}".format(global_step,total_loss,cls_loss,offset_loss,size_loss))
            print("learning_rate:{0:0.6f}".format(lr))
            # print("predicts:", predicts)
            print('The batch run total {0:0.5f}s'.format(time_elapsed))
        batch_num += 1
except tf.errors.OutOfRangeError:
    print('Training has completed...')
train_total_time=(cv2.getTickCount()-train_start_time)/cv2.getTickFrequency()
print('Training has stopped...')
print("Total batch:",batch_num)
hour=train_total_time // 3600
minute=(train_total_time-hour*3600)//60
print('Training runs {:.0f}h {:.0f}m...'.format(hour,minute))
sess.close()

