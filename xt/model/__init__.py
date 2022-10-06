# from .model_creator import model_creator
"""Create model module to define the NN architecture."""

from __future__ import division, print_function

from xt.framework import Registers
from xt.model.model import XTModel
from xt.model.tf_compat import tf
import zeus.common.util.common as common
from xt.model.multi_trainer import MultiTrainerModel


__ALL__ = ['model_builder', 'Model']

def model_builder(model_info):
    """Create the interface func for creating model."""
    model_name = model_info['model_name']
    model_config = model_info.get('model_config', None)
    gpu_nums = model_config.get('gpu_nums', 1)

    if model_info.get('type', 'actor') is 'learner':
        if gpu_nums > 1:
            model = MultiTrainerModel(model_info)
            return model
    model_ = Registers.model[model_name](model_info)
    # print("调用model_builder,得到的model结果不是MultiTrainerModel!!!!!!")
    return model_
