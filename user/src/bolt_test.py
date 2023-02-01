import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras


def bolt_test():

    module_path = "/home/data/cypo/bolt"
    try:
        sys.path.append(module_path)
        import bolt_python_interface as bolt
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "bolt module not found under path:{}".format(module_path))

    # bolt_model_path = "/home/data/dxa/xingtian_revise/impala_compose/user/model/test_model_0_f32.bolt"
    bolt_model_path = "/home/data/dxa/xingtian_revise/impala_compose/user/model/test_model_0_int8_q.bolt"
    inference_batchsize = 2
    action_dim = 4

    bolt_interpreter = bolt.BoltModelWrapper.get_instance()
    bolt_interpreter.prepare(
        bolt_model_path, inference_batchsize, action_dim, 2)

    # update interpreter info
    bolt_interpreter = {
        "interpreter": bolt_interpreter,
        # fixme: experimental revise | pipeline 3 | default 1
        "input_shape": (inference_batchsize, 4, 84, 84),
        "pi_logic_outs_index": 1,
        "baseline_index": 0,
    }

    state = np.random.randn(inference_batchsize, 4, 84, 84)
    # state = np.repeat(state,inference_batchsize,axis=0)
    # state = np.concatenate([state for i in range(inference_batchsize)], axis=0)

    # input_data = np.expand_dims(np.transpose(
    #     np.array(state), (0, 3, 1, 2)).flatten(), 1).astype(np.float32)
    input_data = np.expand_dims(state.astype(np.float32).flatten(), 1)

    input_data2 = np.transpose(
        np.array(state), (0, 2, 3, 1)).astype(np.float32)

    interpreter = bolt_interpreter["interpreter"]
    interpreter.inference(input_data)
    pi_logic_outs = interpreter.get_result(
        bolt_interpreter["pi_logic_outs_index"])
    baseline = interpreter.get_result(
        bolt_interpreter["baseline_index"])

    # time.sleep(0.002)

    p = pi_logic_outs[range(0, inference_batchsize*2, 2)].tolist()
    b = baseline.tolist()

    for pp in p:
        print(pp)
    print("================================================")

    print(b)

    print("================================================")
    print("================================================")

    tflite_model = "/home/data/dxa/xingtian_revise/impala_compose/user/model/test_model_0.tflite"

    interpreter = tf.lite.Interpreter(model_path=tflite_model)
    interpreter.resize_tensor_input(interpreter.get_input_details()[
                                    0]['index'], (inference_batchsize, 84, 84, 4))
    interpreter.allocate_tensors()
    p = interpreter.get_output_details()[0]
    b = interpreter.get_output_details()[1]

    input = interpreter.get_input_details()[0]
    # assert input_data2.shape == (1,84,84,4), input_data2.shape
    interpreter.set_tensor(input['index'], input_data2)
    interpreter.invoke()

    br = interpreter.get_tensor(p['index'])
    pr = interpreter.get_tensor(b['index'])

    for pp in pr:
        print(pp)
    print("================================================")
    print(br)


if __name__ == "__main__":
    bolt_test()
