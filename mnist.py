#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

import dataset
# from models.conv2_dense2_dropout import Model
from models.dense3 import Model

from helpers.gpu_utils import validate_batch_size_for_multi_gpu
from helpers.softmax_cross_entropy_trainer import create_model_fn


def main(_):

    model_function = create_model_fn(lambda params: Model(params), tf.train.AdamOptimizer(learning_rate=1e-4))

    if FLAGS.multi_gpu:
        validate_batch_size_for_multi_gpu(FLAGS.batch_size)

        # There are two steps required if using multi-GPU: (1) wrap the model_fn,
        # and (2) wrap the optimizer. The first happens here, and (2) happens
        # in the model_fn itself when the optimizer is defined.
        model_function = tf.contrib.estimator.replicate_model_fn(
            model_function, loss_reduction=tf.losses.Reduction.MEAN)

    data_format = FLAGS.data_format
    if data_format is None:
        data_format = ('channels_first'
                       if tf.test.is_built_with_cuda() else 'channels_last')

    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=FLAGS.model_dir,
        params={
            'data_format': data_format,
            'multi_gpu': FLAGS.multi_gpu
        })

    # Train the model
    def train_input_fn():
        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes use less memory. MNIST is a small
        # enough dataset that we can easily shuffle the full epoch.
        ds = dataset.training_dataset(FLAGS.data_dir)
        ds = ds.cache().shuffle(buffer_size=50000).\
            batch(FLAGS.batch_size).\
            repeat(FLAGS.train_epochs)
        return ds

    # Set up training hook that logs the training accuracy every 100 steps.
    tensors_to_log = {'train_accuracy': 'train_accuracy'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)
    mnist_classifier.train(input_fn=train_input_fn, hooks=[logging_hook])

    # Evaluate the model and print results
    def eval_input_fn():
        return dataset.test_dataset(FLAGS.data_dir).batch(
            FLAGS.batch_size).make_one_shot_iterator().get_next()

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print()
    print('Evaluation results:\n\t%s' % eval_results)

    # Export the model
    if FLAGS.export_dir is not None:
        image = tf.placeholder(tf.float32, [None, 28, 28])
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'image': image,
        })
        mnist_classifier.export_savedmodel(FLAGS.export_dir, input_fn)


class MNISTArgParser(argparse.ArgumentParser):

    def __init__(self):
        super(MNISTArgParser, self).__init__()

        self.add_argument(
            '--multi_gpu', action='store_true',
            help='If set, run across all available GPUs.')
        self.add_argument(
            '--batch_size',
            type=int,
            default=256,
            help='Number of images to process in a batch')
        self.add_argument(
            '--data_dir',
            type=str,
            default='/var/ellie/data/mnist_fashion',
            help='Path to directory containing the MNIST dataset')
        self.add_argument(
            '--model_dir',
            type=str,
            default='/tmp/mnist_model',
            help='The directory where the model will be stored.')
        self.add_argument(
            '--train_epochs',
            type=int,
            default=1,
            help='Number of epochs to train.')
        self.add_argument(
            '--data_format',
            type=str,
            default=None,
            choices=['channels_first', 'channels_last'],
            help='A flag to override the data format used in the model. '
                 'channels_first provides a performance boost on GPU but is not always '
                 'compatible with CPU. If left unspecified, the data format will be '
                 'chosen automatically based on whether TensorFlow was built for CPU or '
                 'GPU.')
        self.add_argument(
            '--export_dir',
            type=str,
            help='The directory where the exported SavedModel will be stored.')


if __name__ == '__main__':
    parser = MNISTArgParser()
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
