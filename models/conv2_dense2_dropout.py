import tensorflow as tf


class Model:
    """Model to recognize digits in the MNIST dataset.

    Network structure is equivalent to:
    https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/mnist/mnist_deep.py
    and
    https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py

    But written as a tf.keras.Model using the tf.layers API.
    """

    def __init__(self, params):
        """Creates a model for classifying a hand-written digit.

        Args:
          params: Parameter dictionary. Must contain data_format: Either 'channels_first' or 'channels_last'.
            'channels_first' is typically faster on GPUs while 'channels_last' is
            typically faster on CPUs. See
            https://www.tensorflow.org/performance/performance_guide#data_formats
        """

        data_format = params['data_format']
        if data_format == 'channels_first':
            self._input_shape = [-1, 1, 28, 28]
        else:
            assert data_format == 'channels_last'
            self._input_shape = [-1, 28, 28, 1]

        self.conv1 = tf.layers.Conv2D(
            32, 5, padding='same', data_format=data_format, activation=tf.nn.relu)
        self.conv2 = tf.layers.Conv2D(
            64, 5, padding='same', data_format=data_format, activation=tf.nn.relu)
        self.fc1 = tf.layers.Dense(1024, activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(10)
        self.dropout = tf.layers.Dropout(0.4)
        self.max_pool2d = tf.layers.MaxPooling2D(
            (2, 2), (2, 2), padding='same', data_format=data_format)

    def __call__(self, inputs, training):
        """Add operations to classify a batch of input images.

        Args:
          inputs: A Tensor representing a batch of input images.
          training: A boolean. Set to True to add operations required only when
            training the classifier.

        Returns:
          A logits Tensor with shape [<batch_size>, 10].
        """
        y = tf.reshape(inputs, self._input_shape)
        y = self.conv1(y)
        y = self.max_pool2d(y)
        y = self.conv2(y)
        y = self.max_pool2d(y)
        y = tf.layers.flatten(y)
        y = self.fc1(y)
        y = self.dropout(y, training=training)
        return self.fc2(y)
