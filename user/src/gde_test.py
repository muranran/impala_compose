import graph_def_editor as gde
import numpy as np
from tensorflow.compat.v1 import Graph, GraphDef, import_graph_def, Session
from tensorflow.compat.v1.gfile import GFile
import tensorflow as tf
from typing import List
import os

DEFAULT_TFL_NAME = "model.tflite"


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


if __name__ == '__main__':
    # main()
    # resize_shape_test()
    cal_actions()
