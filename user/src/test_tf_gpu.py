import tensorflow as tf
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    print(os.environ)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    print(gpus)
    # with tf.device('/gpu:0'):
    #     gpu_a = tf.random.normal([10000, 1000])
    #     gpu_b = tf.random.normal([1000, 2000])
    #     gpu_c = tf.matmul(gpu_a, gpu_b)
    graph = tf.Graph()
    with graph.as_default():
        a = tf.placeholder(dtype=tf.float32, shape=(None, 4), name="a")
        b = tf.placeholder(dtype=tf.float32, shape=(None, 4), name="b")
        c = a + b

    in1 = np.random.randn(4, 4).astype(np.float32)
    in2 = np.random.randn(4, 4).astype(np.float32)

    with tf.Session(graph=graph).as_default() as sess:

        res = sess.run([c], feed_dict={"a:0": in1, "b:0": in2})
        print(res)


if __name__ == '__main__':
    main()
