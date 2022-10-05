import argparse
import time

import tensorflow as tf
from kungfu.python import current_cluster_size, current_rank, init_from_config
from kungfu.tensorflow.ops import all_reduce


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--rank', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    config = {
        'cluster': {
            'peers': [
                '127.0.0.1:10010',
                '127.0.0.1:10011',
            ],
        },
        'self': {
            'rank': args.rank,
        },
    }

    init_from_config(config)

    rank = current_rank()
    size = current_cluster_size()
    print('%d/%d' % (rank, size))
    a = tf.random_normal([10, 10], dtype=tf.float32)
    b = tf.random_normal([10, 10], dtype=tf.float32)
    c = tf.add(a, b)

    config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(c)
    time.sleep(10)
    y = all_reduce(c)
    time.sleep(10)
    print(c, y)
    print('done')


if __name__ == '__main__':
    main()
