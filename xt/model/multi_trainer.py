import os
import glob
import numpy as np

from xt.model.tf_compat import tf, K, get_sess_graph
from copy import deepcopy
from multiprocessing import Queue, Process
from time import time, sleep
from zeus.common.ipc.uni_comm import UniComm
from zeus.common.ipc.message import message
from zeus.common.util.common import bytes_to_str, check_port
from xt.framework import Registers
from xt.model.model import XTModel
import setproctitle
# import shared_numpy as snp


os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["KUNGFU_ALLREDUCE_STRATEGY"] = "BINARY_TREE_STAR"
os.environ["KUNGFU_CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class MultiTrainerModel(object):
    """
    Model Base class for model module.
    Owing to the same name to Keras.Model, set `XTModel` as the base class.
    User could inherit the XTModel, to implement their model.
    """

    def __init__(self, model_info):
        """
        Initialize XingTian Model.
        To avoid the compatibility problems about tensorflow's versions.
        Model class will hold their graph&session within itself.
        Now, we used the keras's API to create models.
        :param model_info:
        """

        model_name = model_info['model_name']
        model_config = model_info.get('model_config', None)
        self.gpu_nums = model_config.get('gpu_nums', 1)
        sample_batch_step = model_config.get("sample_batch_step", None)
        if sample_batch_step is not None:
            model_config.update(
                {"sample_batch_step": sample_batch_step//self.gpu_nums})

        batch_size = model_config.get('BATCH_SIZE', 200)
        model_config.update({"BATCH_SIZE": batch_size//self.gpu_nums})

        self.trainer_q = create_multi_trainer(self.gpu_nums, model_info)

        model_config.update({'gpu_nums': 1})

        self.model_ = Registers.model[model_name](model_info)

        self.model_name = model_name
        self.first = True

    def create_model(self, model_info):
        """Abstract method for creating model."""
        return self.model_.create_model(model_info)

    def predict(self, state):
        """
        Do predict use the newest model.
        :param state:
        :return: output tensor ref to policy.model
        """
        return self.model_.predict(state)

    def train(self, state, label):
        """Train the model."""
        T0 = time()
        state_split, label_split = send_train_data(
            state, label, self.gpu_nums, self.trainer_q, self.first, model_name=self.model_name)
        T1 = time()
        loss = self.model_.train(state_split, label_split)
        T2 = time()
        # print("Training time: {:.2f}_{:.2f}".format(
        #     (T1-T0)*1000, (T2-T1)*1000))
        if self.first:
            self.first = False
        return loss

    def set_weights(self, weights):
        """Set weight with memory tensor."""
        return self.model_.set_weights(weights)

    def get_weights(self):
        """Get the weights."""
        return self.model_.get_weights()

    def get_grad(self, data):
        return self.model_.get_grad(data)

    def save_model(self, file_name):
        """Save weights into .h5 file."""
        # check max model file to keep
        return self.model_.save_model(file_name)

    def load_model(self, model_name):
        return self.model_.load_model(model_name)

    def save_keras_model(self, ):
        return self.model_.save_keras_model()


class MultiTrainer(object):
    def __init__(self, trainer_id, config_info, reply_q, device_id):
        self.config_info = deepcopy(config_info)
        self.trainer_id = trainer_id
        self.request_q = UniComm("ShareByPlasma")

        # shared_memory
        # self.request_q = snp.Queue()
        self.reply_q = reply_q
        self.device_id = device_id

    def process(self):
        # first = True
        # recv_data = None
        while True:
            # print("接收数据=====================================")

            # if first:
            recv_data = self.request_q.recv()
            #     first = False
            # else:
            #     yes = self.request_q.recv()
            # if first:
            #     state = self.request_q.get()
            #     label = []
            #     [label.append(self.request_q.get()) for i in range(4)]
            #     first = False

            # ready_to_train = self.request_q.get()
            # print("成功接收数据=====================================")
            # if recv_data is None:
            #     raise ValueError("Empty training data in work 2")
            ctr_info, data = recv_data
            if ctr_info['cmd'] in self.process_fn.keys():
                proc_fn = self.process_fn[ctr_info['cmd']]
                proc_fn(data)
            else:
                raise KeyError("invalib cmd: {}".format(ctr_info['cmd']))

            # proc_fn = self.process_fn['trainer']
            # proc_fn({'state': state, 'label': label})

            # # shared_memory
            # state.close()
            # state.unlink()

            # [l.close() for l in label]
            # [l.unlink() for l in label]

    def train(self, recv_data):
        state = recv_data['state']
        label = recv_data['label']
        # print("state.shape: =======================", state.shape)

        # 暂时修改
        loss = self.model.train(state, label)

    def start(self):
        from xt.model import model_builder
        from kungfu.python import init_from_config

        setproctitle.setproctitle("multi_trainer")
        gpu_config = self.config_info.get('gpu_config', None)
        gpu_self = gpu_config.get('self', None)

        # print("self.trainer_id =============== {}".format(self.trainer_id))
        # self.trainer_id = 1
        gpu_self.update({'rank': self.trainer_id})

        self.device_id = 0
        os.environ["KUNGFU_CUDA_VISIBLE_DEVICES"] = str(self.device_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
        init_from_config(gpu_config)
        self.model = model_builder(self.config_info)
        # print("self.config_info: ============", self.config_info)
        self.process_fn = {'trainer': self.train}

        self.process()


def create_multi_trainer(gpu_nums, model_info):
    # 初始化kf模块 并返回trainer_q[request_q]数组 穿件gpu_num-1个multi_trainer
    model_config = model_info.get('model_config', None)
    gpu_node_config = model_info.get('gpu_node_config', ['127.0.0.1'])
    gpu_num = model_info.get('gpu_node_nums', [gpu_nums])

    port = 10015
    gpu_port = []
    for j in range(len(gpu_node_config)):
        for i in range(gpu_num[j]):
            while check_port(gpu_node_config[j], str(port)):
                port += 1
            gpu_port.append(gpu_node_config[j] + ':' + str(port))
            port += 1
    # print("gpu_port ============ {}".format(gpu_port))

    gpu_config = model_info.get('gpu_config', None)
    cluster = gpu_config.get('cluster', None)
    cluster.update({'peers': gpu_port})

    trainer_q = []
    for i in range(gpu_nums):
        if i == 0:
            continue
        else:
            print("i==================={}".format(i))
            trainer_q.append(create_trainer(i, model_info, 0))

    from kungfu.python import init_from_config

    gpu_self = gpu_config.get('self', None)
    gpu_self.update({'rank': 0})
    os.environ["KUNGFU_CUDA_VISIBLE_DEVICES"] = str(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    init_from_config(gpu_config)

    return trainer_q


def create_trainer(trainer_id, model_info, device_id):
    # 创建multitrainer并开启训练 self.model_.train(request_q),并返回request_q
    trainer_info = deepcopy(model_info)
    model_config = trainer_info.get('model_config', None)
    model_config.update({'gpu_nums': 1})
    reply_q = Queue()
    trainer = MultiTrainer(trainer_id, trainer_info, reply_q, device_id)

    # 构建模型并训练
    p = Process(target=trainer.start)

    p.start()
    return trainer.request_q


def send_catpole_impala_data(state, label, gpu_nums, trainer_q):
    state_split = [[] for i in range(gpu_nums)]
    label_split = [[] for i in range(gpu_nums)]
    shape = state[0].shape[0] // gpu_nums

    for i in range(gpu_nums):
        state_split[i].append(state[0][shape*i:shape*(i+1)])
        state_split[i].append(state[1][shape*i:shape*(i+1)])

        label_split[i].append(label[0][shape*i:shape*(i+1)])
        label_split[i].append(label[1][shape*i:shape*(i+1)])

    for i in range(gpu_nums-1):
        train_data = {'state': state_split[i+1], 'label': label_split[i+1]}
        train_msg = message(train_data, cmd="trainer")
        trainer_q[i].send(train_msg)
        
    return state_split[0],label_split[0]


def send_train_data(state, label, gpu_nums, trainer_q, first, **kwargs):
    model_name = kwargs.get("model_name")
    if model_name == "ImpalaMlpLite":
        return send_catpole_impala_data(state, label, gpu_nums, trainer_q)

    list_wrapper = None
    if isinstance(state, list) and len(state) == 1:
        list_wrapper = True

    if list_wrapper:
        state = state[0]

    if gpu_nums == 2:
        t0 = time()

        shape = state.shape[0] // 2
        state0 = state[:shape]
        label0 = [l[:shape] for l in label]

        state1 = state[shape:]
        label1 = [l[shape:] for l in label]

        train_data = {'state': [state1]
                      if list_wrapper else state1, 'label': label1}
        train_msg = message(train_data, cmd="trainer")
        t2 = time()

        trainer_q[0].send(train_msg)

        t3 = time()

        return [state0] if list_wrapper else state0, label0
    elif gpu_nums >= 3:
        t0 = time()

        shape = state.shape[0] // gpu_nums
        state_split = []
        label_split = []
        for i in range(gpu_nums):
            state_split.append(state[shape*i:shape*(i+1)])
            label_split.append([l[shape*i:shape*(i+1)] for l in label])

        for i in range(gpu_nums-1):
            train_data = {'state': [state_split[i+1]]
                          if list_wrapper else state_split[i+1], 'label': label_split[i+1]}
            train_msg = message(train_data, cmd="trainer")
            trainer_q[i].send(train_msg)

        return [state_split[0]] if list_wrapper else state_split[0], label_split[0]

    else:
        raise NotImplementedError


def syn_init_model(sess):
    from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
    sess.run(BroadcastGlobalVariablesOp())
    return sess


def allreduce_optimizer(lr, function, with_keras=False, **kwargs):
    from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
    from kungfu.python import current_cluster_size
    optimizer = function(learning_rate=lr * current_cluster_size(), **kwargs)
    if with_keras:
        optimizer = SynchronousSGDOptimizer(
            optimizer, with_keras=True, nccl=True)
    else:
        optimizer = SynchronousSGDOptimizer(optimizer, nccl=False)
    return optimizer
