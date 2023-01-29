import os
from kungfu.python import init_from_config
from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
import tensorflow as tf
import numpy as np
import multiprocessing as multip
# Build model...


class Model(object):
    def __init__(self):
        self.create_model()

    def create_model(self):
        input_size = 28 * 28
        num_classes = 10
        opt = tf.train.AdamOptimizer(0.01)
        opt = SynchronousSGDOptimizer(opt)
        # create a placeholder for the input
        self.x = tf.placeholder(tf.float32, [None, input_size])
        # add a dense layer
        self.y = tf.keras.layers.Dense(
            num_classes, activation=tf.nn.softmax)(self.x)

        # create a placeholder for the true labels
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        # use cross entropy for the loss
        self.cross_entropy = - \
            tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1])
        self.loss = tf.reduce_mean(self.cross_entropy)
        # minimise the loss
        self.train_op = opt.minimize(self.loss)

        # calculate the number of correctly classified datapoints
        correct_prediction = tf.equal(
            tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        test_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, i):
        print("training process...{}".format(i))

        config = {
            "cluster": {
                "peers": ['127.0.0.1:20015', '127.0.0.1:20016']
            },

            "self": {
                "rank": i
            }
        }
        self.device_id = i

        # init_from_config(config)

        os.environ["KUNGFU_CUDA_VISIBLE_DEVICES"] = str(self.device_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)

        data_x = np.random.randn(100, 28*28)
        data_y = np.random.randn(100, 10)

        # Train your model
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # KungFu Step 2: ensure distributed workers start with consistent states
            # from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
            # sess.run(BroadcastGlobalVariablesOp())

            for step in range(10):
                l, _ = sess.run([self.loss, self.train_op], feed_dict={self.x: np.expand_dims(
                    data_x[step], 0), self.y_: np.expand_dims(data_y[step], 0)})
                print(l)

    def multicases(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
    
def start_kungfu(i):
    config = {
        "cluster": {
            "peers": ['127.0.0.1:20015', '127.0.0.1:20016']
        },

        "self": {
            "rank": i
        }
    }

    init_from_config(config)


def main(size=2):

    tasks = [Model() for i in range(size)]
    # kungfu_procs = [multip.Process(
    #     target=start_kungfu, args=(i,)) for i in range(size)]

    procs = [multip.Process(target=t.train, args=(i,))
             for i, t in enumerate(tasks)]

    pids = [p.pid for p in procs]

    # procs.extend(kungfu_procs)

    for p in procs:
        p.start()

    for p in procs:
        p.join()

    print("pids of training:\n{}".format(pids))
    return pids

def mc():
    model = Model()
    model.multicases()

if __name__ == "__main__":
    # main()
    mc()
    pass
