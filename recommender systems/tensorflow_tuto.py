import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# get the version

print(tf.__version__)  # 2.13.0

# I - Initialization of TensorFlow
# manual initialization
x = tf.constant(25, shape=(1, 1), dtype=tf.float32)
y = tf.constant([[1, 4, 2], [3, 5, 6]])

print(x)
print(y)

# initialization using the shape
x = tf.ones((2, 2))  # create a 2*2 matrix filled with 1
x = tf.zeros((2, 2))  # create a 2*2 matrix filled with 0
x = tf.eye(2)  # create a 2*2 identity matrix
print(x)

# from uniform distribution

x = tf.random.normal((3, 3), mean=0, stddev=1)  # normal centrée reduite
print(x)
x = tf.random.uniform((1, 3), minval=0, maxval=1)
print(x)
# vector from range
x = tf.range(9)
print(x)
# or
x = tf.range(start=1, limit=10, delta=2)
print(x)

# casting to different types from int32 to float64
x = tf.cast(x, dtype=tf.float64)
print(x)
# Mathematical Operations

x = tf.constant([1, 2, 3])
y = tf.constant([3, 4, 6])

z = tf.add(x, y)  # or  z = x + y
print(z)
w = tf.subtract(x, y)
print(w)
print(tf.divide(x, y))
print(tf.multiply(x, y))
print(tf.tensordot(x, y, axes=1))
print(tf.reduce_sum(x * y, axis=0))

# Indexing
