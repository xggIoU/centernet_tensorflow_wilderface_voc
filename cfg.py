# tfrecords_path="/mnt/westerndatadisk/datasets/pascal_voc/voc_train_512x512.tfrecords"
tfrecords_path="/mnt/westerndatadisk/datasets/wilder-face/WIDER_train_512x512.tfrecords"


num_classes=1  #face:1   voc:20
input_image_size=512
down_ratio=4.0
featuremap_h=128
featuremap_w=128
feature_channels=256
lambda_size=0.1
lambda_offset=1.0
l2_weight=0.01


mean_coco= [0.408, 0.447, 0.470]
std_coco= [0.289, 0.274, 0.278]
mean_voc = [0.485, 0.456, 0.406]
std_voc  = [0.229, 0.224, 0.225]

batch_size=14
epochs=20
optimizer='adam'#momentum,rmsprop,adam,adadelta,sgd,ftr,adagradDA,adagrad,ProximalAdagrad,ProximalGrad
lr_type="exponential"# "exponential","fixed","piecewise"
lr=0.0001
lr_value=0.0001
lr_decay_rate= 0.90
lr_decay_steps= 1200
momentum=0.9
lr_boundaries=[2000,5000,15000,20000]
lr_values=[0.1,0.01,0.001,0.0001,0.00001]