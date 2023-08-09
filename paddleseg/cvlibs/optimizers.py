# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle

from paddleseg.utils import logger

'''
optimizer:
  type: AdamW
  weight_decay: 0.01
  decay:
    type: backbone_head_decay
    lr_mult: 0.1
'''

class NaiveOptimizer(object):
    def __init__(self, **args):
        assert 'opt' in args, "optimizer config module should have `opt`"
        self.args = args
    
    @staticmethod
    def _params(model):
        return model.parameters()

    def _optimizer(lr_scheduler, params, **args):
        if optimizer_type == 'sgd':
            return paddle.optimizer.Momentum(lr_scheduler, parameters=params, **args)
        elif optimizer_type == 'adam':
            return paddle.optimizer.Adam(lr_scheduler, parameters=params, **args)
        elif optimizer_type in paddle.optimizer.__all__:
            return getattr(paddle.optimizer, optimizer_type)(lr_scheduler,
                                                                parameters=params,
                                                                **args)
        else:
            raise RuntimeError('Unknown optimizer type {}.'.format(optimizer_type))

    def create_optimizer(self, lr_scheduler, model):
        optimizer_type = self.args.pop('opt')
        params = self._params(model)

        

def optimizer_factory(args, lr_scheduler, model):
    optimizer_type = args.pop('type')

    if 
    params = model.parameters()
    if 'backbone_lr_mult' in args:
        if not hasattr(model, 'backbone'):
            logger.warning('The backbone_lr_mult is not effective because'
                            ' the model does not have backbone')
        else:
            backbone_lr_mult = args.pop('backbone_lr_mult')
            backbone_params = model.backbone.parameters()
            backbone_params_id = [id(x) for x in backbone_params]
            other_params = [
                x for x in params if id(x) not in backbone_params_id
            ]
            params = [{
                'params': backbone_params,
                'learning_rate': backbone_lr_mult
            }, {
                'params': other_params
            }]

    

def backbone_head_decay(model, lr_mult=1.0):
    if not hasattr(model, 'backbone'):
        logger.warning('The backbone_head_decay is invalid because '
                        'the model does not have backbone.')
        params = model.parameters()
    else:
        backbone_params = model.backbone.parameters()
        backbone_params_id = [id(x) for x in backbone_params]
        other_params = [
            x for x in params if id(x) not in backbone_params_id
        ]
        params = [{
            'params': backbone_params,
            'learning_rate': lr_mult
        }, {
            'params': other_params
        }]
    return params

def layer_decay():
    pass