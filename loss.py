import tensorflow as tf
def focal_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch,h,w,c)
      gt_regr (batch,h,w,c)
  '''
  pos_inds = tf.cast(tf.equal(gt,1.0),dtype=tf.float32)
  neg_inds = 1.0-pos_inds
  neg_weights = tf.pow(1.0 - gt, 4.0)
 
  pred=tf.clip_by_value(pred, 1e-6, 1.0 - 1e-6)
  pos_loss = tf.log(pred) * tf.pow(1.0 - pred, 2.0) * pos_inds
  neg_loss = tf.log(1.0 - pred) * tf.pow(pred, 2.0) * neg_weights * neg_inds

  num_pos  = tf.reduce_sum(pos_inds)
  pos_loss = tf.reduce_sum(pos_loss)
  neg_loss = tf.reduce_sum(neg_loss)

   loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def reg_l1_loss(pred, gt):
  '''
  :param pred: (batch,h,w,c)
  :param gt: (batch,h,w,c)
  :return:
  '''
  mask = tf.cast(tf.greater(gt, 0.0), dtype=tf.float32)
  num_pos = (tf.reduce_sum(mask) + tf.convert_to_tensor(1e-4))
  loss = tf.abs(pred - gt) * mask
  loss = tf.reduce_sum(loss) / num_pos
  return loss
