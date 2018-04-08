import tensorflow as tf
import numpy as np


def beta(n, num_bins, l, r, alpha, epsilon):
    """
    return the bin function for the n-th bin of N bins for numbers between l and r.
    alpha and epsilon are stretch and margin parameters
    """
    d = (r - l) / num_bins
    l_n = l + (n-1) * d

    def _beta(x):
        sigmoid_left = tf.nn.sigmoid(alpha * (x-l_n+epsilon))
        sigmoid_right = tf.nn.sigmoid(alpha*(-x+l_n+d+epsilon))
        return sigmoid_left * sigmoid_right

    return _beta


def part_funs(distr, num_bins, alpha):
    """
    Compute the array of partitions functions that count the number of values
    in each of the N partitions of the discrete distribution 'distr'
    """
    d_max = tf.reduce_max(distr)
    d_min = tf.reduce_min(distr)
    d = (d_max - d_min) / (num_bins - 1)
    d_min = d_min - d / 2
    d_max = d_max + d / 2

    return [beta(k, num_bins, d_min, d_max, alpha, epsilon=0) for k in range(1, num_bins + 1)]


def entropy(distr, n, alpha):
    """
    Calculate the entropy of the discrete distribution 'distr' using N partitions
    """
    m = tf.cast(tf.size(distr), dtype=tf.float32)

    partitions = part_funs(distr, n, alpha)
    c = [tf.reduce_sum(partitions[k](distr)) for k in range(n)]  # this is counting the partitions

    sums = tf.reduce_sum(c * tf.log(c))  # this is the sum over all partitions
    return tf.log(m) - sums / m


def cond_entropy(distr, labels, n_classes, n_bins, alpha):
    """
        calculate the conditional entropy of distribution 'distr' given labels 'labels'
    """

    betas = part_funs(distr, n_bins, alpha)

    m = tf.cast(tf.size(distr), dtype=tf.float32)

    # The mask is used to count by class
    mask = [tf.cast(tf.equal(c, labels), tf.float32) for c in range(n_classes)]
    # N_classes lists of counts => N_classes conditional distributions
    cond_counts = np.array([
        [
            tf.reduce_sum(betas[n](distr) * mask[k]) for n in range(n_bins)
        ] for k in range(n_classes)
    ])

    pxy = cond_counts / m                        # p(x,y)
    px = tf.reduce_sum(cond_counts, axis=1) / m  # p(x)

    epsilon_stab = 1e-4  # to stabilize the log term
    # need to transpose forth and back to enable broadcasting
    pxy_px = tf.transpose(tf.transpose(pxy) / (tf.transpose(px)) + epsilon_stab)

    return -tf.reduce_sum(pxy * tf.log(pxy_px))
