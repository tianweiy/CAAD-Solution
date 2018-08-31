
"""Implementation of sample defense.

This defense loads inception resnet v2 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import random

import numpy as np
from scipy.misc import imread

import tensorflow as tf

#import inception_resnet_v2
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

#tf.flags.DEFINE_string(
#    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens3_adv_inception_v3', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens4_adv_inception_v3', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens_adv_inception_resnet_v2', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_adv_inception_v3', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_resnet', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

tf.flags.DEFINE_integer(
    'image_resize', 331, 'Resize of image size.')

FLAGS = tf.flags.FLAGS



def padding_layer_iyswim(inputs, shape, name=None):
    h_start = shape[0]
    w_start = shape[1]
    output_short = shape[2]
    input_shape = tf.shape(inputs)
    input_short = tf.reduce_min(input_shape[1:3])
    input_long = tf.reduce_max(input_shape[1:3])
    output_long = tf.to_int32(tf.ceil(
        1. * tf.to_float(output_short) * tf.to_float(input_long) / tf.to_float(input_short)))
    output_height = tf.to_int32(input_shape[1] >= input_shape[2]) * output_long +\
        tf.to_int32(input_shape[1] < input_shape[2]) * output_short
    output_width = tf.to_int32(input_shape[1] >= input_shape[2]) * output_short +\
        tf.to_int32(input_shape[1] < input_shape[2]) * output_long
    return tf.pad(inputs, tf.to_int32(tf.stack([[0, 0], [h_start, output_height - h_start - input_shape[1]], [w_start, output_width - w_start - input_shape[2]], [0, 0]])), name=name)


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath) as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

#def graph(padded_input,num_classes=1001,i):

#def stop(padded_input,num_classes=1001,i):
 # num_iter = FLAGS.num_iter
  #return tf.less(i, num_iter)

def main(_):
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001
    itr = 30

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        img_resize_tensor = tf.placeholder(tf.int32, [2])
        x_input_resize = tf.image.resize_images(x_input, img_resize_tensor, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        shape_tensor = tf.placeholder(tf.int32, [3])
        padded_input = padding_layer_iyswim(x_input_resize, shape_tensor)
        # 330 is the last value to keep 8*8 output, 362 is the last value to keep 9*9 output, stride = 32
        padded_input.set_shape(
            (FLAGS.batch_size, FLAGS.image_resize, FLAGS.image_resize, 3))

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
            padded_input, num_classes=num_classes, is_training=False, create_aux_logits=True)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_adv_v3, end_points_v3 = inception_v3.inception_v3(
            padded_input, num_classes=num_classes, is_training=False, scope='AdvInceptionV3')

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
            padded_input, num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens3_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
            padded_input, num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits_resnet, end_points_resnet = resnet_v2.resnet_v2_101(
            padded_input, num_classes=num_classes, is_training=False)

        logits=(logits_ensadv_res_v2+logits_adv_v3+logits_ens3_adv_v3+logits_ens3_adv_v3+logits_resnet)/5
        Aux=(end_points_ensadv_res_v2['AuxLogits']+end_points_v3['AuxLogits']+end_points_ens3_adv_v3['AuxLogits']+end_points_ens4_adv_v3['AuxLogits'])*0.1

        predicted_labels = tf.argmax((logits+Aux),1)

        #predicted_labels = tf.argmax(end_points['Predictions'], 1)

        # Run computation
        #saver = tf.train.Saver(slim.get_model_variables())
        #session_creator = tf.train.ChiefSessionCreator(
        #    scaffold=tf.train.Scaffold(saver=saver),
        #    checkpoint_filename_with_path=[FLAGS.checkpoint_path_ens3_adv_inception_v3,FLAGS.checkpoint_path_ens4_adv_inception_v3]
        #    master=FLAGS.master)

        #with tf.train.MonitoredSession(session_creator=session_creator) as sess:

        s1 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        s5 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))







        with tf.Session() as sess:

            s1.restore(sess, FLAGS.checkpoint_path_adv_inception_v3)
            s2.restore(sess, FLAGS.checkpoint_path_ens3_adv_inception_v3)
            s3.restore(sess, FLAGS.checkpoint_path_ens4_adv_inception_v3)
            s4.restore(sess, FLAGS.checkpoint_path_ens_adv_inception_resnet_v2)
            s5.restore(sess, FLAGS.checkpoint_path_resnet)


            with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
                for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                    final_preds = np.zeros(
                        [FLAGS.batch_size, num_classes, itr])
                for j in range(itr):
                    if np.random.randint(0, 2, size=1) == 1:
                        images = images[:, :, ::-1, :]
                        resize_shape_ = np.random.randint(310, 331)


                        final_preds[..., j]= sess.run([predicted_labels],
                                                        feed_dict={x_input: images, img_resize_tensor: [resize_shape_]*2,
                                                                   shape_tensor: np.array([random.randint(0, FLAGS.image_resize - resize_shape_), random.randint(0, FLAGS.image_resize - resize_shape_), FLAGS.image_resize])})

                final_probs = np.sum(final_preds, axis=-1)
                labels = np.argmax(final_probs, 1)

                for filename, label in zip(filenames, labels):
                    out_file.write('{0},{1}\n'.format(filename, label))


if __name__ == '__main__':
    tf.app.run()
