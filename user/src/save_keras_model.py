from tensorflow.compat.v1 import Graph, GraphDef, import_graph_def, Session
from tensorflow.compat.v1.gfile import GFile
import numpy as np
import tensorflow as tf


def main():
    frozen_graph = "/home/tank/dxa/xingtian_revise/impala_opt/user/data/model/imp25.pb"
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

    # set input output
    x = graph.get_tensor_by_name("state_input:0")
    y = graph.get_tensor_by_name("explore_agent/pi_logic_outs:0")
    sess = Session(graph=graph)

    # get batch_input
    batch_image = np.zeros([1, 84, 84, 4])
    # get ...

    # predict
    # feed_dict_testing = {x: batch_image}
    # output = sess.run([y], feed_dict=feed_dict_testing)

    converter = tf.lite.TFLiteConverter.from_session(sess, [x], [y])
    tflite_model = converter.convert()


    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    # print(output)
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)


def read_name():
    name_file = "/home/tank/dxa/xingtian_revise/impala_opt/user/model/model_node_name.txt"
    name_file2 = "/home/tank/dxa/xingtian_revise/impala_opt/user/model/model_node_name2.txt"
    with open(name_file, "r") as f:
        raw_name = f.read()
    names = list(filter(lambda x: x, list(raw_name.split("\n"))))
    print(names)
    with open(name_file2, "w") as f:
        f.write("\[")
        for name in names:
            f.write("\"{}\",".format(name))
        f.write("\]")

def save_as_tflite():
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file="/home/tank/dxa/xingtian_revise/impala_opt/user/data/model/imp25.pb",
        input_arrays=["state_input"],
        output_arrays=["explore_agent/pi_logic_outs"]
    )
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with tf.io.gfile.GFile('/home/tank/dxa/xingtian_revise/impala_opt/user/data/model/imp25.tflite', 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    # read_name()
    main()
    # save_as_tflite()