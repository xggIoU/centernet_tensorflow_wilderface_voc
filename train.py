import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
import cv2
import cfg

from shufflenetv2_centernet_V2 import ShuffleNetV2_centernet
# from shufflenetv2_centernet_V2_SEB import Shufflenetv2_Centernet_SEB
# from yolov3_centernet_V2 import yolov3_centernet
from create_label import CreatGroundTruth


def parse_color_data(example_proto):
    features = {"img_raw": tf.FixedLenFeature([], tf.string),
                "label": tf.FixedLenFeature([], tf.string),
                "width": tf.FixedLenFeature([], tf.int64),
                "height": tf.FixedLenFeature([], tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    img = parsed_features["img_raw"]
    img = tf.decode_raw(img, tf.uint8)
    width = parsed_features["width"]
    height = parsed_features["height"]
    img = tf.reshape(img, [height, width, 3])
    img = tf.cast(img, tf.float32) * (1. / 255.) - 0.5
    label = parsed_features["label"]
    label = tf.decode_raw(label, tf.float32)

    return img, label
def erase_invalid_val(sequence):
    label = []
    h, w = sequence.shape
    mask = (sequence != -1.0)
    for i in range(h):
        seq_new = sequence[i][mask[i]]
        label.append(list(seq_new))
    return label

filenames = [cfg.tfrecords_path]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.map(parse_color_data)
val1=tf.constant(-0.5,tf.float32)
val2 = tf.constant(-1, tf.float32)
dataset = dataset.padded_batch(cfg.batch_size, padded_shapes=([None, None, 3], [None]), padding_values=(val1, val2))
dataset = dataset.repeat(cfg.epochs)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

train_start_time = cv2.getTickCount()


model=ShuffleNetV2_centernet()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter("shufflenetv2_voc_summary", sess.graph)

saver=tf.train.Saver(max_to_keep=20)

if 0:#reload model
    model_file = tf.train.latest_checkpoint('shufflenetv2_voc/')
    saver.restore(sess, model_file)
    print("reload ckpt from "+model_file)
try:
    while True:
        batch_start_time=cv2.getTickCount()
        img_batch, label_batch = sess.run(next_element)
        label_batch = erase_invalid_val(label_batch)
        cls_gt_batch, size_gt_batch = CreatGroundTruth(label_batch)
        feed = {model.inputs: img_batch,
                model.is_training:True,
                model.size_gt:size_gt_batch,
                model.cls_gt:cls_gt_batch
               }
        fetches = [
                   model.cls_loss,
                   model.size_loss,
                   model.total_loss,
                   model.global_step,
                   model.lr,
                   model.merged_summay,
                   model.train_op,
                   ]
        cls_loss,size_loss,total_loss,global_step, lr, summary, _ = sess.run(fetches, feed)

        train_writer.add_summary(summary, global_step)
        time_elapsed = (cv2.getTickCount()-batch_start_time)/cv2.getTickFrequency()

        if global_step%200==0:
            saver.save(sess,"shufflenetv2_seb_voc/shufflenetv2_seb_voc.ckpt",global_step=global_step)
            # saver.save(sess,"shufflenetv2_face_SEB_summary/shufflenetv2_face_SEB.ckpt",global_step=global_step)
            # saver.save(sess,"shufflenetv2_voc/shufflenetv2_voc.ckpt",global_step=global_step)
            # saver.save(sess,"yolov3_voc/yolov3_voc.ckpt",global_step=global_step)
            # saver.save(sess,"shufflenev2_face_ori/shufflenev2_face.ckpt",global_step=global_step)

        if global_step % 10 == 0:
            print("-------Training {0}th batch-------".format(global_step))
            print("global_step:{0} total_loss:{1:0.3f} cls_loss:{2:0.3f} size_loss:{3:0.3f}".format(global_step,total_loss,cls_loss,size_loss))
            print("learning_rate:{0:0.6f}".format(lr))
            # print("predicts:", predicts)
            print('The batch run total {0:0.5f}s'.format(time_elapsed))

except tf.errors.OutOfRangeError:
    print('Training has completed...')
train_total_time=(cv2.getTickCount()-train_start_time)/cv2.getTickFrequency()
print('Training has stopped...')
hour=train_total_time // 3600
minute=(train_total_time-hour*3600)//60
print('Training runs {:.0f}h {:.0f}m...'.format(hour,minute))
sess.close()

