import zmq
from multiprocessing import Process
from subprocess import Popen, PIPE
import shlex
from time import sleep, time


def func(name, *args, **kwargs):
    pass


def main():
    testfiledir = "/home/data/dxa/xingtian_revise/impala_compose/user/model"
    compress_tool_file = "user/src/compress_tool.py"
    weight_file = testfiledir+"/weights.h5"
    tflite_model_file = testfiledir+"/test_model.tflite"
    
    # start compressing
    sp = Popen(shlex.split("/home/tank/miniconda3/envs/tf2/bin/python {}".format(
        compress_tool_file)), shell=False,stdout=PIPE, stderr=PIPE)  # nvidia-smi dmon
    
    sleep(0.5)
    
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    #  Do 10 requests, waiting each time for a response
    for request in range(10):
        message = weight_file + " " + tflite_model_file
        socket.send(message.encode())

        #  Get the reply.
        
        message = socket.recv().decode()
        print("tflite: ",message)
    
    socket.send(b'terminated')
    # sp.terminate()
    print("wait for terminal")
    sleep(0.5)
    print("status: ",sp.poll())
    # sp.kill()
    sp.wait()
    result = sp.stdout.read().decode()+sp.stderr.read().decode()
    print(result)

    pass


if __name__ == '__main__':
    main()
