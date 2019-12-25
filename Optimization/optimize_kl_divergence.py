
# https://towardsdatascience.com/kl-divergence-python-example-b87069e4b810
import numpy as np
from scipy.stats import norm

import tensorflow as tf

import matplotlib.pyplot as plt

"""
To begin, we create a probability distribution with a known mean (0) 
and variance (2). Then, we create another distribution with random 
parameters."""

x = np.arange(-10, 10, 0.001)
p_pdf = norm.pdf(x, 0, 2).reshape(1, -1)
np.random.seed(0)

random_mean = np.random.randint(10, size=1)
random_sig = np.random.randint(10, size=1)
random_pdf = norm.pdf

learning_rate = 0.001
epochs = 100

p = tf.placeholder(tf.float64, shape=p_pdf.shape)
mu = tf.Variable(np.zeros(1))
sigma = tf.Variable(np.eye(1))
normal = tf.exp(-tf.square(x-mu)/(2*sigma))
q = normal / tf.reduce_sum(normal)

kl_divergence = tf.reduce_sum(
    tf.where(p==0, tf.zeros(p_pdf.shape, tf.float64), p*tf.log(p/q))
)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(kl_divergence)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    history = []
    means = []
    variances = []

    for i in range(epochs):
        sess.run(optimizer, {p:p_pdf})
        if i % 10 == 0:
            history.append(sess.run(kl_divergence, {p:p_pdf}))
            means.append(sess.run(mu)[0])
            variances.append(sess.run(sigma)[0][0])
    for mean, variance in zip(means, variances):
        q_pdf = norm.pdf(x, mean, np.sqrt(variance))
        plt.plot(x, q_pdf.reshape(-1, 1), c='red')

    plt.title("KL(P||Q) = %1.3f" % history[-1])
    plt.plot(x, p_pdf.reshape(-1, 1), linewidth=3)
    plt.show()

    plt.plot(history)
    plt.show()

    sess.close()