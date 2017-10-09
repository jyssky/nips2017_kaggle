"""Implementation of sample defense.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import numpy as np
from scipy.misc import imread

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

import inception_resnet_v2

from collections import Counter

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

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

tf.flags.DEFINE_string(
    'temp_output_file_dir', '', 'Temporary output file to save labels.')

FLAGS = tf.flags.FLAGS


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


def session_run_base_inception_v3(num_classes, batch_shape, output_file_name):
    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
                x_input, num_classes=num_classes, is_training=False)

        predicted_labels = tf.argmax(end_points['Predictions'], 1)

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path="inception_v3.ckpt",
            master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            with tf.gfile.Open(output_file_name, 'w') as out_file:
                for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                    labels = sess.run(predicted_labels, feed_dict={x_input: images})
                    for filename, label in zip(filenames, labels):
                        out_file.write('{0},{1}\n'.format(filename, label))


def session_run_base_adv_inception_v3(num_classes, batch_shape, output_file_name):
    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
                x_input, num_classes=num_classes, is_training=False)

        predicted_labels = tf.argmax(end_points['Predictions'], 1)

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path="adv_inception_v3.ckpt",
            master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            with tf.gfile.Open(output_file_name, 'w') as out_file:
                for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                    labels = sess.run(predicted_labels, feed_dict={x_input: images})
                    for filename, label in zip(filenames, labels):
                        out_file.write('{0},{1}\n'.format(filename, label))


def session_run_ens_adv_inception(num_classes, batch_shape, output_file_name):
    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=num_classes, is_training=False)

        predicted_labels = tf.argmax(end_points['Predictions'], 1)

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path="ens_adv_inception_resnet_v2.ckpt",
            master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            with tf.gfile.Open(output_file_name, 'w') as out_file:
                for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                    labels = sess.run(predicted_labels, feed_dict={x_input: images})
                    for filename, label in zip(filenames, labels):
                        out_file.write('{0},{1}\n'.format(filename, label))


def merge_results(result_list):
    def read_temporary_results(temp_output_file):
        with open(temp_output_file) as f:
            map_output = {}
            reader = csv.reader(f)
            for row in reader:
                map_output[row[0]] = row[1]
            return map_output

    final_output_result = {}
    result_maps = [read_temporary_results(x) for x in result_list]
    print(result_maps)
    with open(FLAGS.output_file, 'w') as f:
        writer = csv.writer(f)
        for image_id in result_maps[0].keys():
            cnt = Counter()
            for result_map in result_maps:
                print(result_map[image_id])
                cnt[result_map[image_id]] += 1
            final_output_result[image_id] = cnt.most_common(1)[0][0]
            print("{}.{}.{}".format(cnt, cnt.most_common(1), cnt.most_common(1)[0][0]))
            writer.writerow([image_id, cnt.most_common(1)[0][0]])


def main(_):
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    num_classes = 1001

    print("FLAGS.temp_output_file_dir is {}".format(FLAGS.temp_output_file_dir))

    # Run Inception V3
    output_results_path_inception_v3 = os.path.join(FLAGS.temp_output_file_dir, "inception_v3.csv")
    session_run_base_inception_v3(num_classes, batch_shape, output_results_path_inception_v3)

    # Run adv_inception_v3
    output_results_path_adv_inception_v3 = os.path.join(FLAGS.temp_output_file_dir, "adv_inception_v3.csv")
    session_run_base_adv_inception_v3(num_classes, batch_shape, output_results_path_adv_inception_v3)

    # Run session_run_ens_adv_inception
    output_results_path_ens_adv_inception = os.path.join(FLAGS.temp_output_file_dir, "ens_adv_inception.csv")
    session_run_ens_adv_inception(num_classes, batch_shape, output_results_path_ens_adv_inception)

    output_results = [output_results_path_inception_v3, output_results_path_adv_inception_v3,
                      output_results_path_ens_adv_inception]

    merge_results(output_results)

    # output_results_path_inception_v3 = FLAGS.output_file


if __name__ == '__main__':
    tf.app.run()
