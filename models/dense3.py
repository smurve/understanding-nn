import tensorflow as tf


class Model:
    """Model to recognize digits in the MNIST dataset.

        Super-Naive 3-layer feed forward network
    """

    def __init__(self, params):
        """Creates a model for classifying a hand-written digit.

        Args:
          params: Parameter dictionary. Must contain data_format: Either 'channels_first' or 'channels_last'.
            'channels_first' is typically faster on GPUs while 'channels_last' is
            typically faster on CPUs. See
            https://www.tensorflow.org/performance/performance_guide#data_formats
        """

        self._input_shape = [-1, 784]

        self.fc1 = tf.layers.Dense(1024, activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(128, activation=tf.nn.relu)
        self.fc3 = tf.layers.Dense(10)

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
        y = self.fc1(y)
        y = self.fc2(y)
        return self.fc3(y)
