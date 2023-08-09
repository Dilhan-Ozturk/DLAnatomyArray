import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers



def save_infer_model(model, shape, save_path):
    model.eval()
    model = paddle.jit.to_static(
        model,
        input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32')])
    save_path = os.path.join(save_path, 'model')
    paddle.jit.save(model, save_path)


if __name__ == '__main__':
    