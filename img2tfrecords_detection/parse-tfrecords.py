import numpy as np
import tensorflow as tf
import cv2

def parse_data(example_proto):
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
    # img = tf.cast(img,tf.float32) * (1./255.) - 0.5
    label = parsed_features["label"]
    label=tf.decode_raw(label,tf.float32)
    return img,label
def erase_invalid_val(sequence):
    label=[]
    for seq in sequence:
        seq_new=seq[seq!=-1]
        label.append(list(seq_new))
    return label

if __name__=='__main__':
    tfrecords = "G:/asdsa/voc_train_512x512_2.tfrecords"
    filenames = [tfrecords]
    dataset = tf.data.TFRecordDataset(filenames)

    dataset = dataset.map(parse_data,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(100)
    # dataset = dataset.repeat(2)
    # dataset = dataset.batch(2)
    val1=tf.constant(0,tf.uint8)
    val2 = tf.constant(-1.0, tf.float32)
    dataset = dataset.padded_batch(4, padded_shapes=([None,None,3],[None]),padding_values=(val1,val2))
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    sess = tf.Session()
    num=0
    try:
        while True:
            img, label = sess.run(next_element)
            print(img.shape)
            print(label.shape)
            label=erase_invalid_val(label)#erase -1 in label
            # print(label)
            for j in range(len(label)):
                for i in range(len(label[j])):
                    if i%5==0:
                        cv2.rectangle(img[j],(int(label[j][i+1]),int(label[j][i+2])),(int(label[j][i+3]),int(label[j][i+4])),(0,255,0))
                cv2.imshow("pic",img[j])
                cv2.waitKey()
            num+=1

    except tf.errors.OutOfRangeError:
        print('End:total image',num)

    sess.close()
