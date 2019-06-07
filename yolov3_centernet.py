
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cfg
import loss


from yolov3_layer_utils import conv2d, darknet53_body, yolo_block, upsample_layer


class yolov3_centernet(object):

    def __init__(self):
        self.inputs = tf.placeholder(shape=[None, cfg.input_image_size, cfg.input_image_size, 3], dtype=tf.float32,
                                     name='inputs')
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        self.center_gt = tf.placeholder(shape=[None, cfg.featuremap_h, cfg.featuremap_w, cfg.num_classes],
                                        dtype=tf.float32)
        self.offset_gt = tf.placeholder(shape=[None, cfg.featuremap_h, cfg.featuremap_w, 2], dtype=tf.float32)
        self.size_gt = tf.placeholder(shape=[None, cfg.featuremap_h, cfg.featuremap_w, 2], dtype=tf.float32)
        self.mask_gt = tf.placeholder(shape=[None, cfg.featuremap_h, cfg.featuremap_w], dtype=tf.float32)
        with tf.variable_scope('yolo3_centernet'):
                self.pred_center, self.pred_offset, self.pred_size = self._build_model()
        self.build_train()
        self.merged_summay = tf.summary.merge_all()

    def _build_model(self):
        with slim.arg_scope([slim.batch_norm],is_training=self.is_training):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1)):
                with tf.variable_scope('darknet53_body'):
                    route_1, route_2, route_3 = darknet53_body(self.inputs)

                with tf.variable_scope('yolov3_head'):
                    inter1, net = yolo_block(route_3, 512)  # 13*13*1024->(13*13*512,13*13*1024)
                    inter1 = conv2d(inter1, 256, 1)  # 13*13*512->13*13*256
                    inter1 = upsample_layer(inter1, route_2.get_shape().as_list())  # 26*26*256
                    concat1 = tf.concat([inter1, route_2], axis=3)  # 26*26*(256+512)=26*26*768
                    inter2, net = yolo_block(concat1, 256)  # 26*26*768->(26*26*256,26*26*512)
                    inter2 = conv2d(inter2, 128, 1)  # 26*26*256->26*26*128
                    inter2 = upsample_layer(inter2, route_1.get_shape().as_list())  # 26*26*128->52*52*128
                    concat2 = tf.concat([inter2, route_1], axis=3)  # 52*52*(128+256)->52*52*384
                    _, feature_map_3 = yolo_block(concat2, 128)  # 52*52*384->(52*52*128,52*52*256)

            with tf.variable_scope('detector'):
                center = slim.conv2d(feature_map_3, cfg.feature_channels, 3, 1, padding='same',)
                center = slim.conv2d(center, cfg.num_classes, 1, 1, padding='same',normalizer_fn=None,activation_fn=tf.nn.sigmoid, biases_initializer=tf.zeros_initializer())


                offset = slim.conv2d(feature_map_3, cfg.feature_channels, 3, 1, padding='same',)
                offset = slim.conv2d(offset, 2, 1, 1, padding='same',normalizer_fn=None,activation_fn=None, biases_initializer=tf.zeros_initializer())

                size = slim.conv2d(feature_map_3, cfg.feature_channels, 3, 1, padding='same', )
                size = slim.conv2d(size, 2, 1, 1, padding='same', normalizer_fn=None,
                                     activation_fn=None, biases_initializer=tf.zeros_initializer())
        return center, offset, size
    def compute_loss(self):
        self.cls_loss=loss.focal_loss(self.pred_center,self.center_gt)
        self.size_loss=loss._reg_l1loss(self.pred_size,self.size_gt,self.mask_gt)
        self.offset_loss = loss._reg_l1loss(self.pred_offset, self.offset_gt, self.mask_gt)
        # self.regular_loss=cfg.weight_decay * tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        self.total_loss=self.cls_loss+cfg.lambda_size*self.size_loss+cfg.lambda_offset*self.offset_loss#+self.regular_loss

    def build_train(self):
        with tf.variable_scope("loss","loss"):
            self.compute_loss()
            self.global_step = tf.Variable(0, trainable=False)
            self.lr=cfg.lr
            if cfg.lr_type=="exponential":
                self.lr = tf.train.exponential_decay(cfg.lr_value,
                                                     self.global_step,
                                                     cfg.lr_decay_steps,
                                                     cfg.lr_decay_rate,
                                                     staircase=True)#staircase=True,globstep/decaystep=整数，代表lr突变的，阶梯状
            elif cfg.lr_type=="fixed":
                self.lr = tf.constant(cfg.lr, dtype=tf.float32)
            elif cfg.lr_type=="piecewise":
                self.lr = tf.train.piecewise_constant(self.global_step, cfg.lr_boundaries, cfg.lr_values)
            if cfg.optimizer == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            elif cfg.optimizer == 'rmsprop':
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr,
                                                           momentum=cfg.momentum)
            elif cfg.optimizer == 'adadelta':
                self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
            elif cfg.optimizer == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr,
                                                            momentum=cfg.momentum)
            elif cfg.optimizer=="sgd":
                self.optimizer=tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            elif cfg.optimizer == "ftr":
                self.optimizer = tf.train.FtrlOptimizer(learning_rate=self.lr)
            elif cfg.optimizer == "adagradDA":
                self.optimizer = tf.train.AdagradDAOptimizer(learning_rate=self.lr, global_step=self.global_step)
            elif cfg.optimizer == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
            elif cfg.optimizer == "ProximalAdagrad":
                self.optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=self.lr)
            elif cfg.optimizer == "ProximalGrad":
                self.optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=self.lr)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)

        tf.summary.scalar('total_loss', self.total_loss)
        tf.summary.scalar('cls_loss', self.cls_loss)
        tf.summary.scalar('offset_loss', self.offset_loss)
        tf.summary.scalar('size_loss', self.size_loss)
        tf.summary.scalar("learning_rate", self.lr)

