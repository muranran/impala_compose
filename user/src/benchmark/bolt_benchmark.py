import os
import sys

from time import time, sleep
import numpy as np
import tensorflow as tf
from tensorflow import keras
from multiprocessing import Process, Queue
import psutil

# use cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def keras_cpu_perf(model_path):
    # common record
    data_store = []
    infer_time = []
    pure_infer_time = []

    sleep(2)  # waiting bind core if needed

    # create test data
    keras_data = np.random.randn(1, 84, 84, 4).astype(np.float32)
    # keras_data = np.ones(shape=(1, 84, 84, 4)).astype(np.float32)
    # keras_data = np.zeros(shape=(1, 84, 84, 4)).astype(np.float32)
    bolt_data = np.transpose(keras_data, (0, 3, 1, 2))
    bolt_data = np.expand_dims(bolt_data.flatten(), 1)

    # init recorder
    infer_time.clear()
    pure_infer_time.clear()
    result = None

    # create keras model
    start_0 = time()
    model = keras.models.load_model(model_path)
    t = time() - start_0

    # run inference
    for i in range(2000):
        start_0 = time()
        result = model(keras_data)
        infer_time.append(time() - start_0)
    print("prepare model time:{:.2f}ms\nmean infer time fp32:{:.2f}ms\n".format(
        t * 1000,
        np.mean(infer_time) * 1000))

    # print result
    for rest in result:
        print(rest.numpy())


def bolt_cpu_perf(model_path, module_path):
    # import bolt infer module
    try:
        sys.path.append(module_path)
        import example
    except ModuleNotFoundError as err:
        print("bolt module not found under path:{}".format(module_path))

    # common record
    data_store = []
    infer_time = []
    pure_infer_time = []

    sleep(2)  # waiting bind core if needed
    # init recorder
    infer_time.clear()
    pure_infer_time.clear()
    result = np.random.randn(5, 1)

    # create bolt model
    s2 = example.Dog.get_instance()
    start_0 = time()
    s2.prepare(model_path)
    t = time() - start_0

    # create test data
    keras_data = np.random.randn(1, 84, 84, 4).astype(np.float32)
    # keras_data = np.ones(shape=(1, 84, 84, 4)).astype(np.float32)
    # keras_data = np.zeros(shape=(1, 84, 84, 4)).astype(np.float32)
    bolt_data = np.transpose(keras_data, (0, 3, 1, 2))
    bolt_data = np.expand_dims(bolt_data.flatten(), 1)

    # run inference
    for i in range(2000):
        start_0 = time()
        pt = s2.inference(bolt_data, result)
        infer_time.append(time() - start_0)
        pure_infer_time.append(pt)

    # print result
    print("prepare model time:{:.2f}ms\nmean infer time fp32:{:.2f}ms\npure infer time:{:.2f}ms | {:.2f}%\n".format(
        t * 1000,
        np.mean(infer_time) * 1000,
        np.mean(pure_infer_time),
        np.mean(pure_infer_time) / (np.mean(infer_time) * 1000) * 100))


if __name__ == '__main__':
    # core bind
    core_used = [5]

    # bolt python module path
    module_path = './'

    # model path
    model_keras = './test.h5'
    model_fp32 = './ppo_cnn_f32.bolt'
    model_int8 = './ppo_cnn_int8_q.bolt'

    # # test keras cpu perf
    # print("===============keras result===============")
    # p = Process(target=keras_cpu_perf, args=(model_keras,))
    # p.start()
    # psutil.Process(p.pid).cpu_affinity([5])
    # p.join()

    # test bolt fp32 perf
    print("===============bolt fp32 result===============")
    p = Process(target=bolt_cpu_perf, args=(model_fp32, module_path,))
    p.start()
    psutil.Process(p.pid).cpu_affinity([5])
    p.join()

    # test bolt int8 perf
    print("===============bolt int8 result===============")
    p = Process(target=bolt_cpu_perf, args=(model_int8, module_path,))
    p.start()
    psutil.Process(p.pid).cpu_affinity([5])
    p.join()
