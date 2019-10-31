

import tensorflow as tf

tf.enable_eager_execution()

x = tf.ones((2, 2))
print x

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

dz_dx = t.gradient(z, x)
for i in [0,1]:
    for j in [0,1]:
        print dz_dx[i,j]