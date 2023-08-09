# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.cvlibs import manager
from paddleseg.models import layers


@manager.MODELS.add_component
class AnatomyUNet(nn.Layer):


    def __init__(self,
                 num_classes,
                 align_corners=False,
                 use_deconv=False,
                 pretrained=None):
        super().__init__()

        self.encode = Encoder()
        self.aspp = ASPP(in_channels=512, out_channels=256, atrous_rates=[1, 6, 12, 18])
        self.decode = Decoder(align_corners, use_deconv=use_deconv)
        self.cls = self.conv = nn.Conv2D(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1)

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        logit_list = []
        x, short_cuts = self.encode(x)
        x = self.aspp(x)
        x = self.decode(x, short_cuts)
        logit = self.cls(x)
        logit_list.append(logit)
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class Encoder(nn.Layer):
    def __init__(self):
        super().__init__()

        self.double_conv = nn.Sequential(
            layers.ConvBNReLU(3, 64, 3), layers.ConvBNReLU(64, 64, 3))
        down_channels = [[64, 128], [128, 256], [256, 512], [512, 512]]
        self.down_sample_list = nn.LayerList([
            self.down_sampling(channel[0], channel[1])
            for channel in down_channels
        ])

        self.se_blocks = nn.LayerList([
            SELayer(channel[1]) for channel in down_channels
        ])

    def down_sampling(self, in_channels, out_channels):
        modules = []
        modules.append(nn.MaxPool2D(kernel_size=2, stride=2))
        modules.append(layers.ConvBNReLU(in_channels, out_channels, 3))
        modules.append(layers.ConvBNReLU(out_channels, out_channels, 3))
        return nn.Sequential(*modules)

    def forward(self, x):
        short_cuts = []
        x = self.double_conv(x)
        for down_sample, se_blocks in zip(self.down_sample_list, self.se_blocks):
            short_cuts.append(x)
            x = down_sample(x)
        return x, short_cuts


class Decoder(nn.Layer):
    def __init__(self, align_corners, use_deconv=False):
        super().__init__()

        up_channels = [[512, 256], [256, 128], [128, 64], [64, 64]]
        self.up_sample_list = nn.LayerList([
            UpSampling(channel[0], channel[1], align_corners, use_deconv)
            for channel in up_channels
        ])

        self.se_blocks = nn.LayerList([
            SELayer(channel[1]) for channel in up_channels
        ])

        self.cbam = nn.LayerList([
            CBAM(channel[1]) for channel in up_channels
        ])

    def forward(self, x, short_cuts):
        for i in range(len(short_cuts)):
            x = self.up_sample_list[i](x, short_cuts[-(i + 1)])
            # x = self.se_blocks[i](x)
            x = self.cbam[i](x)
        return x


class UpSampling(nn.Layer):
    def  __init__(self,
                 in_channels,
                 out_channels,
                 align_corners,
                 use_deconv=False):
        super().__init__()

        self.align_corners = align_corners

        self.use_deconv = use_deconv
        if self.use_deconv:
            self.deconv = nn.Conv2DTranspose(
                in_channels,
                out_channels // 2,
                kernel_size=2,
                stride=2,
                padding=0)
            in_channels = in_channels + out_channels // 2
        else:
            in_channels *= 2

        self.fusion = nn.Conv2D(in_channels, in_channels, 1)

        self.double_conv = nn.Sequential(
            layers.ConvBNReLU(in_channels, out_channels, 3),
            layers.ConvBNReLU(out_channels, out_channels, 3))




    def forward(self, x, short_cut):
        if self.use_deconv:
            x = self.deconv(x)
        else:
            x = F.interpolate(
                x,
                paddle.shape(short_cut)[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        x = paddle.concat([x, short_cut], axis=1)
        x = self.double_conv(x)
        return x

class ASPP(nn.Layer):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super().__init__()
        self.aspp1 = layers.ConvBNReLU(in_channels, out_channels, 1, stride=1, dilation=atrous_rates[0])
        self.aspp2 = layers.ConvBNReLU(in_channels, out_channels, 3, stride=1, dilation=atrous_rates[1], padding=atrous_rates[1])
        self.aspp3 = layers.ConvBNReLU(in_channels, out_channels, 3, stride=1, dilation=atrous_rates[2], padding=atrous_rates[2])
        self.aspp4 = layers.ConvBNReLU(in_channels, out_channels, 3, stride=1, dilation=atrous_rates[3], padding=atrous_rates[3])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2D((1, 1)),
            layers.ConvBNReLU(in_channels, out_channels, 1, stride=1))

        self.conv1 = layers.ConvBNReLU(out_channels*5, out_channels, 1, stride=1)
        self.conv2 = nn.Conv2D(out_channels, 512, 1, stride=1)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=True)

        x = paddle.concat((x1, x2, x3, x4, x5), axis=1)

        x = self.conv1(x)
        x = self.conv2(x)

        return x


class SELayer(nn.Layer):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).reshape((b, c))
        y = self.fc(y).reshape((b, c, 1, 1))
        return x * y

class CBAM(nn.Layer):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.max_pool = nn.AdaptiveMaxPool2D(output_size=1)
        self.fc1 = nn.Conv2D(in_channels=channels, out_channels=channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(in_channels=channels // reduction, out_channels=channels, kernel_size=1, padding=0)

        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2D(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention Module
        module_input = x
        avg = self.relu(self.fc1(self.avg_pool(x)))
        avg = self.fc2(avg)
        mx = self.relu(self.fc1(self.max_pool(x)))
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)

        # Spatial Attention Module
        x = module_input * x
        module_input = x
        avg = paddle.mean(x, axis=1, keepdim=True)
        mx = paddle.argmax(x, axis=1, keepdim=True)
        mx = paddle.cast(mx, 'float32')
        x = paddle.concat([avg, mx], axis=1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x

        return x