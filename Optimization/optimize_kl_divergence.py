
# https://towardsdatascience.com/kl-divergence-python-example-b87069e4b810
import numpy as np
from scipy.stats import norm

import tensorflow as tf

import matplotlib.pyplot as plt

"""
To begin, we create a probability distribution with 
a known mean (0) and variance (2). Then, we create another 
distribution with random parameters."""

x = np.arange(-10, 10, 0.001)
p_pdf = norm.pdf(x, 0, 2).reshape(1, -1) # Target
np.random.seed(0)

random_mean = np.random.randint(10, size=1) # low = 10
random_sig = np.random.randint(10, size=1)
random_pdf = norm.pdf

"""
Given that we are using gradient descent, 
we need to select values for the hyperparameters 
(i.e. step size, number of iterations).
"""

learning_rate = 0.001
epochs = 100


"""
Just like numpy, in tensorflow we need to allocate memory for variables. 
"""
p = tf.placeholder(tf.float64, shape=p_pdf.shape) # utils. like i.

mu = tf.Variable(np.zeros(1))
sigma = tf.Variable(np.eye(1))

normal = tf.exp(-tf.square(x-mu)/(2*sigma)) # util

"""
Computes the sum of elements across dimensions of a tensor.

Reduces `input_tensor` along the dimensions given in `axis`.
"""
q = normal / tf.reduce_sum(normal) # normalized guess

"""
Just like before, we define a function to compute the KL 
divergence that excludes probabilities equal to zero:

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))
"""
kl_divergence = tf.reduce_sum(
    tf.where(p==0, tf.zeros(p_pdf.shape, tf.float64), p*tf.log(p/q))
        # condition, trueVal, falseVal
)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(kl_divergence)

"""
Only after running tf.global_variables_initializer() will the 
variables hold the values we set when we declared them (i.e. tf.zeros).
"""

init = tf.global_variables_initializer()

"""
All operations in tensorflow must be done within a session
"""

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