import sys
import os
import numpy as np
from time import time


def main():
    # import bolt batch infer .so
    module_path = "/home/data/cypo/bolt"
    try:
        sys.path.append(module_path)
        import batch_infer as bolt
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError("batch_infer not found under \"{}\"".format(module_path))

    # Create test cases. Default batch size is 3.
    keras_data = np.random.randn(3, 84, 84, 4).astype(np.float32)
    # keras_data = np.ones(shape=(1, 84, 84, 4)).astype(np.float32)
    # keras_data = np.zeros(shape=(1, 84, 84, 4)).astype(np.float32)
    bolt_data = np.transpose(keras_data, (0, 3, 1, 2))
    bolt_data = np.expand_dims(bolt_data.flatten(), 1)

    # Create batch infer instance
    model_path = "/home/data/dxa/xingtian_revise/impala_opt/user/data/model/model_10_f32.bolt"
    bi = bolt.get_instance()
    bi.prepare(model_path)

    start_0 = time()
    bi.inference(bolt_data)
    res1 = bi.get_result(0)
    res2 = bi.get_result(1)
    infer_time = time() - start_0
    print("infer_time is : {:.2f}ms".format(infer_time*1000))
    print(res1)
    print(res2)


if __name__ == '__main__':
    main()
