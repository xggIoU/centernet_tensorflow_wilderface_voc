tfrecords_path="/mnt/westerndatadisk/datasets/pascal_voc/voc_train_512x512_sort_box_shuffle.tfrecords"
# tfrecords_path="/mnt/westerndatadisk/datasets/wilder-face/WIDER_train_512x512_shuffle.tfrecords"


#(1)shufflenet_voc
num_classes=20
input_image_size=512
down_ratio=4.0
featuremap_h=128
featuremap_w=128
feature_channels=256

#(2)shufflenet_face
# num_classes=1
# input_image_size=512
# down_ratio=4.0
# featuremap_h=128
# featuremap_w=128
# feature_channels=256

#(3)yolov3_voc
# num_classes=20
# input_image_size=512
# down_ratio=8.0
# featuremap_h=64
# featuremap_w=64
# feature_channels=256

batch_size=16
epochs=10
optimizer='adam'#momentum,rmsprop,adam,adadelta,sgd,ftr,adagradDA,adagrad,ProximalAdagrad,ProximalGrad
lr_type="exponential"# "exponential","fixed","piecewise"
lr=0.0001
lr_value=0.00025
lr_decay_rate= 0.85
lr_decay_steps= 2200
momentum=0.9
lr_boundaries=[2000,5000,15000,20000]
lr_values=[0.1,0.01,0.001,0.0001,0.00001]