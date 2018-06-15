import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("Tensorflow version is : " + str(tf.__version__))

hello = tf.constant("Hello from tensorflow")

sess = tf.Session()

print(sess.run(hello))
