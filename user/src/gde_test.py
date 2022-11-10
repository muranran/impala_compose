import os
import time
from typing import List

import graph_def_editor as gde
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import Graph, GraphDef, import_graph_def, Session
from tensorflow.compat.v1.gfile import GFile
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from functools import partial

DEFAULT_TFL_NAME = "model.tflite"
os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)


def save_as_tflite(pb_file: str, input_names: List[str], output_names: List[str], save_path: str):
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file=pb_file,
        input_arrays=input_names,
        output_arrays=output_names
    )
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    if save_path.endswith(".tflite"):
        tflite_file = save_path
    else:
        tflite_file = os.path.join(save_path, DEFAULT_TFL_NAME)
    with tf.io.gfile.GFile(tflite_file, 'wb') as f:
        f.write(tflite_model)

    return tflite_file


def main():
    graph_path = "/home/data/dxa/xingtian_revise/impala_opt/user/data/model/model_0_0.pb"
    # frozen_graph = "/home/tank/dxa/xingtian_revise/impala_opt/user/data/model/imp25.pb"
    frozen_graph = "/home/data/dxa/xingtian_revise/impala_opt/user/data/model/model_0.pb"
    # import graph
    with GFile(frozen_graph, "rb") as f:
        graph_def = GraphDef()
        graph_def.ParseFromString(f.read())
    with Graph().as_default() as graph:
        import_graph_def(graph_def,
                         input_map=None,
                         return_elements=None,
                         name=""
                         )
    g = gde.Graph(graph.as_graph_def())
    # g = gde.saved_model_to_graph(frozen_graph)
    gde.rewrite.change_batch_size(g, new_size=3, inputs=[g["state_input"]])
    # g.to_saved_model(graph_path)
    # graph_revised = g.to_tf_graph()
    graph_revised = g.to_graph_def()
    with Graph().as_default() as graph_r:
        import_graph_def(graph_revised,
                         input_map=None,
                         return_elements=None,
                         name=""
                         )

    # with open(graph_path, "wb") as f:
    #     f.write(graph_r.SerializeToString())
    x = graph_r.get_tensor_by_name("state_input:0")
    y = graph_r.get_tensor_by_name("explore_agent/pi_logic_outs:0")
    z = graph_r.get_tensor_by_name("explore_agent/baseline:0")
    sess = Session(graph=graph_r)
    converter = tf.lite.TFLiteConverter.from_session(sess, [x], [y, z])
    tflite_model = converter.convert()

    # tfl_graph_path = "/home/data/dxa/xingtian_revise/impala_opt/user/data/model/model_0_0.tflite"
    # tfl_file = save_as_tflite(graph_path,
    #                           ["state_input"],
    #                           ["explore_agent/pi_logic_outs", "explore_agent/baseline"],
    #                           tfl_graph_path)
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    # print(output)
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    print("input shape is : ", input_shape)
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)


def resize_shape_test():
    real_batch_size = 8
    batch_size = 3
    state = [np.zeros((1, 1, 1)) for i in range(real_batch_size)]
    if real_batch_size > batch_size:
        state_zero = np.zeros(state[0].shape)
        num_predict = real_batch_size // batch_size
        rest_num = real_batch_size % batch_size
        state_batch_resize = []
        for i in range(num_predict):
            state_batch_resize.append(state[i * batch_size: (i + 1) * batch_size])
        state_batch_resize.append(
            [*state[num_predict * batch_size:], *[state_zero for i in range(batch_size - rest_num)]])
        print(state[num_predict * batch_size:])
        print([state_zero for i in range(batch_size - rest_num)])

        # for s in state_batch_resize:
        #     print(len(s))
        batch_len = [len(s) for s in state_batch_resize]
        print(batch_len, rest_num, num_predict)
        print(state_batch_resize)


def cal_actions():
    pi_logic_outs = []
    for i in range(5):
        pi_logic_outs.append([0.1, 0.2, 0.3, 0.4])
    actions = tf.squeeze(tf.multinomial(pi_logic_outs, num_samples=1, output_dtype=tf.int32))
    result = actions.eval(session=tf.compat.v1.Session())
    print(result)


def categorical_sample(self, logits):
    self.probs = self.softmax(logits)
    a = [i for i in range(len(logits[0]))]
    action_index = np.random.choice(a, p=self.probs[0])
    return action_index


def get_action(self, pi_latent):
    action = self.categorical_sample(pi_latent)
    np_logp = self.log_prob_np(action)
    action = np.expand_dims(action, axis=0)
    np_logp = np.expand_dims(np_logp, axis=0)
    np_logp = np.expand_dims(np_logp, axis=0)
    return action, np_logp


def softmax(self, logits):
    e_x = np.exp(logits)
    probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
    return probs


def cal_actions_with_np():
    pi_logic_outs = []
    for i in range(5):
        pi_logic_outs.append([0.1, 0.2, 0.3, 0.4])
    pi_logic_outs = np.array(pi_logic_outs)

    # pi_logic_outs = [[8.22197151184082, -198.59768676757812, 1084.2288818359375, -1870.9298095703125],
    #                  [5.455597400665283, -200.9584197998047, 1102.4227294921875, -1896.616455078125],
    #                  [3.170361280441284, -201.0675506591797, 1085.162353515625, -1859.453857421875]]

    # actions = tf.squeeze(tf.multinomial(pi_logic_outs, num_samples=1, output_dtype=tf.int32))
    # result = actions.eval(session=tf.compat.v1.Session())
    def softmax(logits):
        e_x = np.exp(logits)
        probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return probs

    t0 = time.time()
    print([softmax(plo) for plo in pi_logic_outs])
    actions = np.array([np.argmax(np.random.multinomial(1, softmax(plo))) for plo in pi_logic_outs])
    # actions = np.random.multinomial(1, pi_logic_outs[0])
    t = time.time() - t0
    print(actions, "{:.2f}ms".format(t * 1000))


def state_transform(x, mean=1e-5, std=255., input_dtype="float32"):
    """Normalize data."""
    if input_dtype in ("float32", "float", "float64"):
        return x

    # only cast non-float32 state
    if np.abs(mean) < 1e-4:
        return tf.cast(x, dtype='float32') / std
    else:
        return (tf.cast(x, dtype="float32") - mean) / std


class NewModel:
    def __init__(self):
        filters_84x84 = [
            [16, 8, 4],
            [32, 4, 2],
            [256, 11, 1],
        ]
        self.sta_mean = 0.
        self.sta_std = 255.
        self.input_dtype = "uint8"
        self.filter_arch = filters_84x84
        self._transform = partial(state_transform,
                                  mean=self.sta_mean,
                                  std=self.sta_std,
                                  input_dtype=self.input_dtype)
        pass

    def create_model(self):
        with tf.variable_scope("explore_agent"):
            state_input = Lambda(self._transform)(self.ph_state)
            last_layer = state_input

            for (out_size, kernel, stride) in self.filter_arch[:-1]:
                last_layer = Conv2D(
                    out_size,
                    (kernel, kernel),
                    strides=(stride, stride),
                    activation="relu",
                    padding="same",
                )(last_layer)

            # last convolution
            (out_size, kernel, stride) = self.filter_arch[-1]
            convolution_layer = Conv2D(
                out_size,
                (kernel, kernel),
                strides=(stride, stride),
                activation="relu",
                padding="valid",
            )(last_layer)

            self.pi_logic_outs = tf.squeeze(
                Conv2D(4, (1, 1), padding="same")(convolution_layer),
                axis=[1, 2],
                name="pi_logic_outs"
            )

            baseline_flat = Flatten()(convolution_layer)
            self.baseline = tf.squeeze(
                tf.layers.dense(
                    inputs=baseline_flat,
                    units=1,
                    activation=None,
                    kernel_initializer=custom_norm_initializer(0.01),
                ),
                1,
                name="baseline",
            )
            self.out_actions = tf.squeeze(
                tf.multinomial(self.pi_logic_outs, num_samples=1, output_dtype=tf.int32),
                1,
                name="out_action",
            )


def gde_revise_input():
    # Create a graph
    tf_g = tf.Graph()
    with tf_g.as_default():
        a = tf.constant(1.0, shape=[2, 3], name="a")
        c = tf.add(
            tf.placeholder(dtype=tf.uint8),
            tf.placeholder(dtype=tf.uint8),
            name="c")

    # Serialize the graph
    g = gde.Graph(tf_g.as_graph_def())

    # Modify the graph.
    # In this case we replace the two input placeholders with constants.
    # One of the constants (a) is a node that was in the original graph.
    # The other one (b) we create here.
    b = gde.make_const(g, "b", np.full([2, 3], 2.0, dtype=np.float32))
    # b = gde.make_placeholder(g, "b", shape=[2, 3], dtype=tf.float32)
    gde.swap_inputs(g[c.op.name], [g[a.name], b.output(0)])

    # Reconstitute the modified serialized graph as TensorFlow graph
    input0 = np.full([2, 3], 2.0, dtype=np.float32)
    with g.to_tf_graph().as_default():
        # Run a session using the modified graph and print the value of c
        with tf.Session() as sess:
            res = sess.run(c.name)
            print("Result is:\n{}".format(res))


def rebuild_graph():
    filters_84x84 = [
        [16, 8, 4],
        [32, 4, 2],
        [256, 11, 1],
    ]


def revise_graph_input_test():
    # a = tf.placeholder(shape=[], dtype=tf.float32, name='a')
    b = tf.placeholder(shape=[], dtype=tf.float32, name='b')
    c = tf.placeholder(shape=[], dtype=tf.float32, name='c')
    a = tf.constant(5.)
    # build graph
    d = a * b

    op = b.consumers()[0]  # <op here is the mul operation of d>
    print(op)
    op._update_input(1, c)  # <update the first input of op with c>

    with tf.Session().as_default() as sess:
        res = sess.run([d], feed_dict={"c:0": 3})
        print(res)

def testn():
    frozen_graph = "/home/data/dxa/xingtian_revise/impala_opt/user/data/model/model_0.pb"
    with GFile(frozen_graph, "rb") as f:
        graph_def = GraphDef()
        graph_def.ParseFromString(f.read())
    with Graph().as_default() as graph:
        import_graph_def(graph_def,
                         input_map=None,
                         return_elements=None,
                         name=""
                         )


if __name__ == '__main__':
    # main()
    # resize_shape_test()
    # cal_actions_with_np()
    # gde_revise_input()
    revise_graph_input_test()