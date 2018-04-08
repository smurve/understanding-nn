"""
    This code is pretty much a hack.
"""

import tensorflow as tf
import numpy as np
import math


def beta(idx, n_bins, d_min, d_max, alpha):
    """
    compute a 2 or 3-dimensional partition function
    :param idx:    the two or three dimensional index
    :param n_bins: the number of partitions
    :param d_min:  array of the left boundaries, one for each dimension
    :param d_max:  array of the right boundaries, one for each dimension
    :param alpha:  stretch factor
    :return: a function that returns a tensor, given an input batch
    """
    d = (d_max - d_min) / n_bins
    l_n = d_min + (idx-1) * d  # the left boundary

    def _beta(x):
        if len(idx) == 2:
            sigmoid_left = tf.nn.sigmoid(alpha * (x - l_n))
            sigmoid_right = tf.nn.sigmoid(alpha * (-x + l_n + d))
            return sigmoid_left[:, 0] * sigmoid_right[:, 0] * \
                sigmoid_left[:, 1] * sigmoid_right[:, 1]
        elif len(idx) == 3:
            sigmoid_left = tf.nn.sigmoid(alpha * (x - l_n))
            sigmoid_right = tf.nn.sigmoid(alpha * (-x + l_n + d))
            return sigmoid_left[:, 0] * sigmoid_right[:, 0] * \
                sigmoid_left[:, 1] * sigmoid_right[:, 1] * \
                sigmoid_left[:, 2] * sigmoid_right[:, 2]

    return _beta


def part_funs2(distr, n_bins, alpha, d_min=None, d_max=None):
    """
    :param distr: a distribution over 2 variables (an array of pairs)
    :param n_bins: the number of bins per variable
    :param alpha: A numerical stretch factor
    :param d_min: array of left bounds, one for each variable
    :param d_max: array of right bounds, one for each variable

    Compute the array of partitions functions that count the number of values
    in each of the N^2 partitions of the discrete distribution 'distr' over
    two variables
    """
    spillover = 1e-2
    # These computations are all on all features

    if d_min is None:
        d_max = tf.reduce_max(distr, axis=0)

    if d_max is None:
        d_min = tf.reduce_min(distr, axis=0)

    d = (d_max - d_min) / n_bins
    d_min = d_min - d * spillover
    d_max = d_max + d * spillover

    sample_dim = distr.shape[1].value
    if sample_dim == 2:
        betas = [beta(np.array((i0, i1)), n_bins, d_min, d_max, alpha)
                 for i0 in range(1, n_bins + 1)
                 for i1 in range(1, n_bins + 1)]
        return betas
    elif sample_dim == 3:
        betas = [beta(np.array((i0, i1, i2)), n_bins, d_min, d_max, alpha)
                 for i0 in range(1, n_bins + 1)
                 for i1 in range(1, n_bins + 1)
                 for i2 in range(1, n_bins + 1)]
        return betas
    else:
        raise ValueError("Can only handle 2 or three dimensional input")


def _entropy(distr, n_bins, alpha, d_min=None, d_max=None):
    res, _ = entropy(distr, n_bins, alpha, d_min, d_max)
    return res


def entropy(distr, n_bins, alpha, d_min=None, d_max=None):
    """
    Calculate the entropy of the discrete distribution 'distr' using n^2 partitions

    :param distr: distribution over two variables (array of pairs)
    :param n_bins: the number of bins for each variable
    :param alpha: stretch factor
    :param d_min: array of left bounds, one for each variable
    :param d_max: array of right bounds, one for each variable

    :return the entropy and the relevant bin counts
    """
    sample_dim = distr.shape[1].value
    n_all_bins = int(math.pow(n_bins, sample_dim))

    m = tf.cast(tf.size(distr) / sample_dim, dtype=tf.float64)

    partitions = part_funs2(distr, n_bins, alpha, d_min, d_max)
    conc_betas = tf.concat(
        tf.transpose(
            [partitions[k](distr) for k in range(n_all_bins)]),
        axis=0)

    epsilon_stab = 1e-8
    c = tf.reduce_sum(conc_betas, axis=0) + epsilon_stab

    sums = tf.reduce_sum(c * tf.log(c))  # this is the sum over all partitions
    return tf.log(m) - sums / m, c


def _cond_entropy(distr, labels, n_classes, n_bins, alpha, d_min=None, d_max=None):
    res, _ = cond_entropy(distr, labels, n_classes, n_bins, alpha, d_min, d_max)
    return res


def cond_entropy(distr, labels, n_classes, n_bins, alpha, d_min=None, d_max=None):
    """
    calculate the conditional entropy of distribution 'distr' given labels 'labels'
    :param distr: distribution over two variables (array of pairs)
    :param labels: labels for distr
    :param n_classes: the number of different labels
    :param n_bins: the number of bins for each variable
    :param alpha: stretch factor
    :param d_min: array of left bounds, one for each variable
    :param d_max: array of right bounds, one for each variable
    """
    sample_dim = distr.shape[1].value
    n_all_bins = int(math.pow(n_bins, sample_dim))

    betas = part_funs2(distr, n_bins, alpha, d_min, d_max)

    m = tf.cast(tf.size(distr) / sample_dim, dtype=tf.float64)

    # The mask is used to count by class
    mask = [tf.cast(tf.equal(c, tf.cast(labels, tf.int32)), tf.float64) for c in range(n_classes)]
    # N_classes lists of counts => N_classes conditional distributions
    cond_counts = [
        [
            tf.reduce_sum(betas[n](distr) * mask[k]) for n in range(n_all_bins)
        ] for k in range(n_classes)
    ]

    pxy = cond_counts / m                        # p(x,y)
    px = tf.reduce_sum(cond_counts, axis=1) / m  # p(x)

    epsilon_stab = 1e-4  # to stabilize the log term
    # need to transpose forth and back to enable broadcasting
    pxy_px = tf.transpose(tf.transpose(pxy) / (tf.transpose(px)) + epsilon_stab)

    return -tf.reduce_sum(pxy * tf.log(pxy_px)), cond_counts


def info_gain(distr, labels, n_classes, n_bins, alpha, d_min=None, d_max=None):
    total_entropy = _entropy(distr, n_bins, alpha, d_min, d_max)
    conditional_entropy = _cond_entropy(distr, labels, n_classes, n_bins, alpha, d_min, d_max)
    return total_entropy - conditional_entropy


def test_stability():
    min_max2 = [np.array([0, 0]), np.array([1,1])]
    sess = tf.InteractiveSession()
    distr = tf.constant([[.7, .2], [.5, .4], [.3, .1], [.5, .4],
                         [.3, .8], [.4, .9], [.9, .2], [.5, .1], ], dtype=tf.float64)
    labels = tf.constant([1, 0, 0, 0, 1, 1, 1, 0])
    res, counts = sess.run(entropy(distr, 2, 1e5, *min_max2))
    print("Entropy: %s, counts: %s" % (res, counts))
    if math.isnan(res):
        raise ValueError("Entropy calculation instable!")
    res, counts = sess.run(cond_entropy(distr, labels, 2, 2, 1e5, *min_max2))
    print("Conditional Entropy: %s, counts: %s" % (res, counts))
    if math.isnan(res):
        raise ValueError("Entropy calculation instable!")
