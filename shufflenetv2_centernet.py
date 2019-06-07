from shufflenetv2_layer_utils import *
import cfg
import loss
class ShuffleNetV2_centernet():

    first_conv_channel = 24

    def __init__(self, model_scale=1.0, shuffle_group=2):
        self.inputs = tf.placeholder(shape=[None, cfg.input_image_size, cfg.input_image_size, 3], dtype=tf.float32, name='inputs')
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        self.shuffle_group = shuffle_group
        self.channel_sizes = self._select_channel_size(model_scale)
        self.center_gt = tf.placeholder(shape=[None, cfg.featuremap_h, cfg.featuremap_w, cfg.num_classes],dtype=tf.float32)
        self.offset_gt = tf.placeholder(shape=[None, cfg.featuremap_h, cfg.featuremap_w, 2], dtype=tf.float32)
        self.size_gt = tf.placeholder(shape=[None, cfg.featuremap_h, cfg.featuremap_w, 2], dtype=tf.float32)
        self.mask_gt = tf.placeholder(shape=[None, cfg.featuremap_h, cfg.featuremap_w], dtype=tf.float32)
        with tf.variable_scope('shufflenet_centernet'):
            with slim.arg_scope([slim.batch_norm], is_training=self.is_training):
                self.pred_center,self.pred_offset,self.pred_size=self._build_model()
        self.build_train()
        self.merged_summay = tf.summary.merge_all()
    def _select_channel_size(self, model_scale):
        # [(out_channel, repeat_times), (out_channel, repeat_times), ...]
        if model_scale == 0.5:
            return [(48, 4), (96, 8), (192, 4), (1024, 1)]
        elif model_scale == 1.0:
            return [(116, 4), (232, 8), (464, 4), (1024, 1)]
        elif model_scale == 1.5:
            return [(176, 4), (352, 8), (704, 4), (1024, 1)]
        elif model_scale == 2.0:
            return [(244, 4), (488, 8), (976, 4), (2048, 1)]
        else:
            raise ValueError('Unsupported model size.')

    def _build_model(self):
        with tf.variable_scope('stage_4'):
            out_2 = conv_bn_relu(self.inputs, self.first_conv_channel, 3, 2)#/2
            out_4 = slim.max_pool2d(out_2, 3, 2 , padding='SAME')#/4

        with tf.variable_scope('stage_8'):
            out_channel, repeat = self.channel_sizes[0]
            # First block is downsampling
            out_8 = shufflenet_v2_block(out_4, out_channel, 3, 2, shuffle_group=self.shuffle_group)#/8
            for i in range(repeat-1):
                out_8 = shufflenet_v2_block(out_8, out_channel, 3, shuffle_group=self.shuffle_group)

        with tf.variable_scope('stage_16'):
            out_channel, repeat = self.channel_sizes[1]
            # First block is downsampling
            out_16 = shufflenet_v2_block(out_8, out_channel, 3, 2, shuffle_group=self.shuffle_group)#/16
            for i in range(repeat - 1):
                out_16 = shufflenet_v2_block(out_16, out_channel, 3, shuffle_group=self.shuffle_group)

        with tf.variable_scope('stage_32'):
            out_channel, repeat = self.channel_sizes[2]
            # First block is downsampling
            out_32 = shufflenet_v2_block(out_16, out_channel, 3, 2, shuffle_group=self.shuffle_group)#/32
            for i in range(repeat - 1):
                out_32 = shufflenet_v2_block(out_32, out_channel, 3, shuffle_group=self.shuffle_group)

        with tf.variable_scope('feature_map_fuse'):
            deconv1=deconv_bn_relu(out_32,cfg.feature_channels)
            out_16=conv_bn_relu(out_16,cfg.feature_channels,1)
            fuse1=deconv1+out_16

            deconv2 = deconv_bn_relu(fuse1, cfg.feature_channels)
            out_8 = conv_bn_relu(out_8, cfg.feature_channels, 1)
            fuse2 = out_8 + deconv2

            deconv3 = deconv_bn_relu(fuse2, cfg.feature_channels)
            out_4 = conv_bn_relu(out_4, cfg.feature_channels, 1)
            fuse3 = out_4 + deconv3

        with tf.variable_scope('se_sa_module'):
            features=se_unit(fuse3)
            features=sa_unit(features)

        with tf.variable_scope('detector'):
            center = tf.layers.conv2d(features, cfg.feature_channels, 3, 1, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.01))
            center = tf.nn.relu(center)
            center = tf.layers.conv2d(center, cfg.num_classes, 1, 1, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.01))
            center = tf.nn.sigmoid(center, name='center')

            offset = tf.layers.conv2d(features, cfg.feature_channels, 3, 1, padding='same',
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01))
            offset = tf.nn.relu(offset)
            offset = tf.layers.conv2d(offset, 2, 1, 1, padding='same',
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01))
            offset = tf.nn.sigmoid(offset, name='offset')
            # size = conv_bn_activation(features,256,3,1)
            size = tf.layers.conv2d(features, cfg.feature_channels, 3, 1, padding='same',
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01))
            size = tf.nn.relu(size)
            size = tf.layers.conv2d(size, 2, 1, 1, padding='same',
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01))
            size = tf.nn.relu(size, name='size')
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