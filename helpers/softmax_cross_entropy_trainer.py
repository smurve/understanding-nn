import tensorflow as tf


def create_model_fn(model_factory, optimizer, dict_key=None):
    """
    Create the model_fn to hand to the Estimator
    :param model_factory: A function that takes the model_fn params to create the model tensor
    :param dict_key: optional key to lookup the input tensor if 'features' is a dictionary
    :param optimizer: the optimizer to use for training, ignored if not running in training mode
    :return:
    """

    def _model_fn(features, labels, mode, params):
        """The model_fn argument for creating an Estimator.

        :param features: the features, either input tensor or dictionary, if dictionary, dict_key is looked up.
        :param labels: the true labels
        :param mode: see: tf.estimator.ModeKeys
        :param params: any params
        :return: an appropriate EstimatorSpec for the chosen mode.
        """
        input_tensor = features
        if isinstance(input_tensor, dict):
            input_tensor = features[dict_key]

        model = model_factory(params)

        if mode == tf.estimator.ModeKeys.PREDICT:
            logits = model(input_tensor, training=False)
            predictions = {
                'classes': tf.argmax(logits, axis=1),
                'probabilities': tf.nn.softmax(logits),
            }
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.PREDICT,
                predictions=predictions,
                export_outputs={
                    'classify': tf.estimator.export.PredictOutput(predictions)
                })

        if mode == tf.estimator.ModeKeys.TRAIN:

            the_optimizer = optimizer
            # If we are running multi-GPU, we need to wrap the optimizer.
            if params.get('multi_gpu'):
                the_optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

            logits = model(input_tensor, training=True)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            accuracy = tf.metrics.accuracy(
                labels=labels, predictions=tf.argmax(logits, axis=1))
            # Name the accuracy tensor 'train_accuracy' to demonstrate the
            # LoggingTensorHook.
            tf.identity(accuracy[1], name='train_accuracy')
            tf.summary.scalar('train_accuracy', accuracy[1])
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.TRAIN,
                loss=loss,
                train_op=the_optimizer.minimize(loss, tf.train.get_or_create_global_step()))

        if mode == tf.estimator.ModeKeys.EVAL:
            logits = model(input_tensor, training=False)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=loss,
                eval_metric_ops={
                    'accuracy':
                        tf.metrics.accuracy(
                            labels=labels,
                            predictions=tf.argmax(logits, axis=1)),
                })
    return _model_fn
