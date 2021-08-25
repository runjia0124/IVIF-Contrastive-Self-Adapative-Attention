import time
import os
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim
from options.train_options import TrainOptions
from torch.utils.tensorboard import SummaryWriter
from lib.nn import SynchronizedBatchNorm2d as SynBN2d
from utils import pad_tensor
from utils import pad_tensor_back
import torchfile
from torchvision import models
from resnet import resnet18
from attention import CAM_Module
from P_loss import Vgg19_Unet, Vgg19_train

class ziji_vgg(nn.Module):
    def __init__(self):
        super(ziji_vgg, self).__init__()

        self.v1_conv2_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.v1_ReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.v1_bn2_1 = nn.BatchNorm2d(64)
        self.v1_max_pool_1 = nn.MaxPool2d(2, 2)
        self.v1_conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.v1_ReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.v1_bn2_2 = nn.BatchNorm2d(64)
        self.v1_max_pool_2 = nn.MaxPool2d(2, 2)

        self.v1_conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.v1_ReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.v1_bn3_1 = nn.BatchNorm2d(128)
        self.v1_conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.v1_ReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.v1_bn3_2 = nn.BatchNorm2d(128)
        self.v1_conv3_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.v1_ReLU3_3 = nn.LeakyReLU(0.2, inplace=True)
        self.v1_bn3_3 = nn.BatchNorm2d(128)
        self.v1_max_pool_3 = nn.MaxPool2d(2, 2)

        self.v1_conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.v1_ReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.v1_bn4_1 = nn.BatchNorm2d(256)
        self.v1_conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.v1_ReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.v1_bn4_2 = nn.BatchNorm2d(256)
        self.v1_conv4_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.v1_ReLU4_3 = nn.LeakyReLU(0.2, inplace=True)
        self.v1_bn4_3 = nn.BatchNorm2d(256)
        self.v1_max_pool_4 = nn.MaxPool2d(2, 2)

    def forward(self, input):
        vis_2 = self.v1_bn2_1(self.v1_ReLU2_1(self.v1_conv2_1(input)))
        vis_2 = self.v1_max_pool_1(vis_2)
        vis_2 = self.v1_bn2_2(self.v1_ReLU2_2(self.v1_conv2_2(vis_2)))
        x = self.v1_max_pool_2(vis_2)

        vis_3 = self.v1_bn3_1(self.v1_ReLU3_1(self.v1_conv3_1(x)))
        vis_3 = self.v1_bn3_2(self.v1_ReLU3_2(self.v1_conv3_2(vis_3)))
        vis_3 = self.v1_bn3_3(self.v1_ReLU3_3(self.v1_conv3_3(vis_3)))
        x = self.v1_max_pool_3(vis_3)

        vis_4 = self.v1_bn4_1(self.v1_ReLU4_1(self.v1_conv4_1(x)))
        vis_4 = self.v1_bn4_2(self.v1_ReLU4_2(self.v1_conv4_2(vis_4)))
        vis_4 = self.v1_bn4_3(self.v1_ReLU4_3(self.v1_conv4_3(vis_4)))
        return vis_2, vis_3, vis_4


class Vgg(nn.Module):
    def __init__(self):
        super(Vgg, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.ReLU1_1 = nn.ReLU()
        self.bn1_1 = nn.BatchNorm2d(64)
        self.max_pool_1_1 = nn.MaxPool2d(2, 2)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.ReLU1_2 = nn.ReLU()
        self.bn1_2 = nn.BatchNorm2d(64)
        self.max_pool_1_2 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.ReLU2_1 = nn.ReLU()
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.ReLU2_2 = nn.ReLU()
        self.bn2_2 = nn.BatchNorm2d(128)
        # self.conv2_3 = nn.Conv2d(128, 128, 3, padding=1)
        # self.ReLU2_3 = nn.ReLU()
        self.max_pool_2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.ReLU3_1 = nn.ReLU()
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.ReLU3_2 = nn.ReLU()
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.ReLU3_3 = nn.ReLU()
        self.bn3_3 = nn.BatchNorm2d(256)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.ReLU3_4 = nn.ReLU()
        self.bn3_4 = nn.BatchNorm2d(256)
        # self.max_pool_3 = nn.MaxPool2d(2, 2)
    def forward(self, input):
        o_64 = self.bn1_1(self.ReLU1_1(self.conv1_1(input)))
        o_64 = self.max_pool_1_1(o_64)
        o_64 = self.bn1_2(self.ReLU1_2(self.conv1_2(o_64)))
        x = self.max_pool_1_2(o_64)

        o_128 = self.bn2_1(self.ReLU2_1(self.conv2_1(x)))
        o_128 = self.bn2_2(self.ReLU2_2(self.conv2_2(o_128)))
        x = self.max_pool_2(o_128)

        o_256 = self.bn3_1(self.ReLU3_1(self.conv3_1(x)))
        o_256 = self.bn3_2(self.ReLU3_2(self.conv3_2(o_256)))
        o_256 = self.bn3_3(self.ReLU3_3(self.conv3_3(o_256)))
        o_256 = self.bn3_4(self.ReLU3_4(self.conv3_4(o_256)))
        return o_64, o_128, o_256


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # print(y.expand_as(x).shape)
        # print(y.expand_as(x))
        return x * y.expand_as(x), y.expand_as(x)


class Unet_resize_conv(nn.Module):
    def __init__(self):
        super(Unet_resize_conv, self).__init__()

        # self.vgg = load_vgg16("./core/Losses/vgg", cfg)
        # self.vgg.to_relu_1_2[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.se_32 = CAM_Module(32)
        self.se_64 = CAM_Module(64)
        self.se_128 = CAM_Module(128)
        self.se_256 = CAM_Module(256)

        self.vgg19 = Vgg19_Unet(vgg19_weights='place_holder')


        self.skip = False
        p = 1
        # self.conv1_1 = nn.Conv2d(4, 32, 3, padding=p)

        self.conv1_1 = nn.Conv2d(2, 32, 3, padding=p)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.max_pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.deconv4 = nn.Conv2d(256 * 3, 256, 3, padding=p)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.max_pool4 = nn.MaxPool2d(2)

        # self.conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
        # self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        # if self.opt.use_norm == 1:
        #     self.bn5_1 = SynBN2d(512) if self.opt.syn_norm else nn.BatchNorm2d(512)
        # self.conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        # self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        # if self.opt.use_norm == 1:
        #     self.bn5_2 = SynBN2d(512) if self.opt.syn_norm else nn.BatchNorm2d(512)

        # self.deconv5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        # self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        # self.conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
        # self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        # if self.opt.use_norm == 1:
        #     self.bn6_1 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)
        # self.conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
        # self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        # if self.opt.use_norm == 1:
        #     self.bn6_2 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)

        # self.deconv6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.att_deconv7 = nn.Conv2d(128*3, 128, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_1 = nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_2 = nn.BatchNorm2d(128)

        # self.deconv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.att_deconv8 = nn.Conv2d(64 * 3, 64, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_1 = nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_2 = nn.BatchNorm2d(64)

        # self.deconv8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.att_deconv9 = nn.Conv2d(32 * 3, 32, 3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn9_1 = nn.BatchNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv10 = nn.Conv2d(32, 1, 1)

        self.tanh = nn.Tanh()

    def forward(self, ir, vis):
        maps = []
        # print(ir.shape)
        # print(vis.shape)

        input = torch.cat([ir, vis], 1)
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        if input.size()[3] > 2200:
            avg = nn.AvgPool2d(2)
            input = avg(input)
            flag = 1
            # pass
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        vis, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(vis)
        ir, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(ir)

        vis_2, vis_3, vis_4 = self.vgg19(vis)
        ir_2, ir_3, ir_4 = self.vgg19(ir)


        flag = 0

        x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
        conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
        x = self.max_pool1(conv1)

        x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
        conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
        x = self.max_pool2(conv2)

        x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
        conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
        x = self.max_pool3(conv3)

        x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))  # 256
        unet_out, unet_map = self.se_256(x)
        vgg_v_out, vgg_v_map = self.se_256(vis_4)
        vgg_i_out, vgg_i_map = self.se_256(ir_4)
        # maps.append(unet_map)
        # maps.append(vgg_v_map)
        # maps.append(vgg_i_map)
        # vis_c4 = torch.cat([unet_out, vgg_v_out], 1)  # 256 + 256
        # ir_c4 = torch.cat([unet_out, vgg_i_out], 1)  # 256 + 256
        att_4 = torch.cat([unet_out, vgg_v_out, vgg_i_out], 1)
        x = self.deconv4(att_4)  # 256*3 -> 256, deconv the concated attention maps
        conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))

        conv6 = F.upsample(conv4, scale_factor=2, mode='bilinear')  # 256
        # print(conv3.shape)
        # print(y_c_features[1].shape)
        # conv3 = conv3 * y_c_features[1] if self.opt.self_attention else conv3
        unet_out, unet_map = self.se_128(conv3) # conv3 128
        vgg_v_out, vgg_v_map = self.se_128(vis_3)
        vgg_i_out, vgg_i_map = self.se_128(ir_3)
        # maps.append(unet_map)
        # maps.append(vgg_v_map)
        # maps.append(vgg_i_map)
        # vis_c7 = torch.cat([unet_out, vgg_v_out], 1)  # 128(conv3) + 128
        # ir_c7 = torch.cat([unet_out, vgg_i_out], 1)  # 128 + 128
        att_7 = torch.cat([unet_out, vgg_v_out, vgg_i_out], 1)
        att_7 = self.att_deconv7(att_7) # 128*3 -> 128
        up7 = torch.cat([self.deconv6(conv6), att_7], 1) # deconv6, 256->128
        # up7 = self.deconv6(torch.cat([conv6, vis_c7, ir_c7], 1))  # 256 + 256 + 256 -> 256
        x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))  #
        conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))  # 128

        unet_out, unet_map = self.se_64(conv2)
        vgg_v_out, vgg_v_map = self.se_64(vis_2)
        vgg_i_out, vgg_i_map = self.se_64(ir_2)
        maps.append(conv2)
        maps.append(unet_out)

        # maps.append(self.se_64(vis_2))
        # maps.append(self.se_64(ir_2))
        conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')
        # vis_c8 = torch.cat([unet_out, vgg_v_out], 1)  # 64(conv2) + 64
        # ir_c8 = torch.cat([unet_out, vgg_i_out], 1)  # 64 + 64
        att_8 = torch.cat([unet_out, vgg_v_out, vgg_i_out], 1)
        att_8 = self.att_deconv8(att_8) # 64*3 -> 64
        # maps.append(att_8)
        up8 = torch.cat([self.deconv7(conv7), att_8], 1) # deconv7, 128->64
        x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
        conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))  # 64

        conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')
        # conv1 = conv1 * gray if self.opt.self_attention else conv1
        # unet_out, unet_map = self.se_32(conv1)
        # vgg_v_out, vgg_v_map = self.se_32(vis_1)
        # vgg_i_out, vgg_i_map = self.se_32(ir_1)
        # maps.append(unet_out)
        # maps.append(conv1)
        # maps.append(vgg_i_out)
        # vis_c9 = torch.cat([unet_out, vgg_v_out], 1)  # 32 + 32
        # ir_c9 = torch.cat([unet_out, vgg_i_out], 1)  # 32 + 32
        # att_9 = torch.cat([unet_out, vgg_v_out, vgg_i_out], 1)
        # att_9 = self.att_deconv9(att_9) # 32*3 -> 32
        up9 = torch.cat([self.deconv8(conv8), conv1], 1) # deconv8, 64 -> 32
        x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
        conv9 = self.LReLU9_2(self.conv9_2(x))

        latent = self.conv10(conv9)

        latent = self.tanh(latent)
        # latent = (latent + 1) / 2
        output = latent

        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        # gray = pad_tensor_back(gray, pad_left, pad_right, pad_top, pad_bottom)
        if flag == 1:
            output = F.upsample(output, scale_factor=2, mode='bilinear')
        if self.skip:
            return output, latent
        else:
            return output

class Unet(nn.Module):
    """
    UNet only, without attention
    """
    def __init__(self):
        super(Unet, self).__init__()

        # self.vgg = load_vgg16("./core/Losses/vgg", cfg)
        # self.vgg.to_relu_1_2[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.se_32 = CAM_Module(32)
        self.se_64 = CAM_Module(64)
        self.se_128 = CAM_Module(128)
        self.se_256 = CAM_Module(256)

        self.skip = False
        p = 1
        # self.conv1_1 = nn.Conv2d(4, 32, 3, padding=p)

        self.conv1_1 = nn.Conv2d(2, 32, 3, padding=p)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.max_pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.deconv4 = nn.Conv2d(256 * 3, 256, 3, padding=p)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.max_pool4 = nn.MaxPool2d(2)

        # self.conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
        # self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        # if self.opt.use_norm == 1:
        #     self.bn5_1 = SynBN2d(512) if self.opt.syn_norm else nn.BatchNorm2d(512)
        # self.conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        # self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        # if self.opt.use_norm == 1:
        #     self.bn5_2 = SynBN2d(512) if self.opt.syn_norm else nn.BatchNorm2d(512)

        # self.deconv5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        # self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        # self.conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
        # self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        # if self.opt.use_norm == 1:
        #     self.bn6_1 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)
        # self.conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
        # self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        # if self.opt.use_norm == 1:
        #     self.bn6_2 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)

        # self.deconv6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.att_deconv7 = nn.Conv2d(128*3, 128, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_1 = nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_2 = nn.BatchNorm2d(128)

        # self.deconv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.att_deconv8 = nn.Conv2d(64 * 3, 64, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_1 = nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_2 = nn.BatchNorm2d(64)

        # self.deconv8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.att_deconv9 = nn.Conv2d(32 * 3, 32, 3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn9_1 = nn.BatchNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv10 = nn.Conv2d(32, 1, 1)

        self.tanh = nn.Tanh()

    def forward(self, ir, vis):
        maps = []
        # print(ir.shape)
        # print(vis.shape)

        input = torch.cat([ir, vis], 1)
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        if input.size()[3] > 2200:
            avg = nn.AvgPool2d(2)
            input = avg(input)
            flag = 1
            # pass
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        # vis, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(vis)
        # ir, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(ir)

        # vis_2, vis_3, vis_4 = self.vgg19(vis)
        # ir_2, ir_3, ir_4 = self.vgg19(ir)


        flag = 0

        x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
        conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
        x = self.max_pool1(conv1)

        x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
        conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
        x = self.max_pool2(conv2)

        x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
        conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
        x = self.max_pool3(conv3)

        x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))  # 256
        conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))
        conv4 = self.se_256(conv4)

        conv6 = F.upsample(conv4, scale_factor=2, mode='bilinear')  # 256
        # print(conv3.shape)
        # print(y_c_features[1].shape)
        # conv3 = conv3 * y_c_features[1] if self.opt.self_attention else conv3
        conv3 = self.se_128(conv3)
        up7 = torch.cat([self.deconv6(conv6), conv3], 1) # deconv6, 256->128
        # up7 = self.deconv6(torch.cat([conv6, vis_c7, ir_c7], 1))  # 256 + 256 + 256 -> 256
        x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))  #
        conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))  # 128
        conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')

        conv2 = self.se_64(conv2)
        up8 = torch.cat([self.deconv7(conv7), conv2], 1) # deconv7, 128->64
        x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
        conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))  # 64

        conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')
        # conv1 = conv1 * gray if self.opt.self_attention else conv1
        # unet_out, unet_map = self.se_32(conv1)
        # vgg_v_out, vgg_v_map = self.se_32(vis_1)
        # vgg_i_out, vgg_i_map = self.se_32(ir_1)
        # maps.append(unet_out)
        # maps.append(conv1)
        # maps.append(vgg_i_out)
        # vis_c9 = torch.cat([unet_out, vgg_v_out], 1)  # 32 + 32
        # ir_c9 = torch.cat([unet_out, vgg_i_out], 1)  # 32 + 32
        # att_9 = torch.cat([unet_out, vgg_v_out, vgg_i_out], 1)
        # att_9 = self.att_deconv9(att_9) # 32*3 -> 32
        conv1 = self.se_32(conv1)
        up9 = torch.cat([self.deconv8(conv8), conv1], 1) # deconv8, 64 -> 32
        x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
        conv9 = self.LReLU9_2(self.conv9_2(x))

        latent = self.conv10(conv9)

        latent = self.tanh(latent)
        # latent = (latent + 1) / 2
        output = latent

        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        # gray = pad_tensor_back(gray, pad_left, pad_right, pad_top, pad_bottom)
        if flag == 1:
            output = F.upsample(output, scale_factor=2, mode='bilinear')
        if self.skip:
            return output, latent
        else:
            return output

class Unet_train(nn.Module):
    """
    UNet with not pretrained vgg
    """
    def __init__(self):
        super(Unet_train, self).__init__()

        # self.vgg = load_vgg16("./core/Losses/vgg", cfg)
        # self.vgg.to_relu_1_2[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.se_32 = CAM_Module(32)
        self.se_64 = CAM_Module(64)
        self.se_128 = CAM_Module(128)
        self.se_256 = CAM_Module(256)

        self.vgg19 = Vgg19_train(requires_grad=True, vgg19_weights='place_holder')


        self.skip = False
        p = 1
        # self.conv1_1 = nn.Conv2d(4, 32, 3, padding=p)

        self.conv1_1 = nn.Conv2d(2, 32, 3, padding=p)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.max_pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.deconv4 = nn.Conv2d(256 * 3, 256, 3, padding=p)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.max_pool4 = nn.MaxPool2d(2)

        # self.conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
        # self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        # if self.opt.use_norm == 1:
        #     self.bn5_1 = SynBN2d(512) if self.opt.syn_norm else nn.BatchNorm2d(512)
        # self.conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        # self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        # if self.opt.use_norm == 1:
        #     self.bn5_2 = SynBN2d(512) if self.opt.syn_norm else nn.BatchNorm2d(512)

        # self.deconv5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        # self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        # self.conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
        # self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        # if self.opt.use_norm == 1:
        #     self.bn6_1 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)
        # self.conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
        # self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        # if self.opt.use_norm == 1:
        #     self.bn6_2 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)

        # self.deconv6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.att_deconv7 = nn.Conv2d(128*3, 128, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_1 = nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_2 = nn.BatchNorm2d(128)

        # self.deconv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.att_deconv8 = nn.Conv2d(64 * 3, 64, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_1 = nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_2 = nn.BatchNorm2d(64)

        # self.deconv8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.att_deconv9 = nn.Conv2d(32 * 3, 32, 3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn9_1 = nn.BatchNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv10 = nn.Conv2d(32, 1, 1)

        self.tanh = nn.Tanh()

    def forward(self, ir, vis):
        maps = []
        # print(ir.shape)
        # print(vis.shape)

        input = torch.cat([ir, vis], 1)
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        if input.size()[3] > 2200:
            avg = nn.AvgPool2d(2)
            input = avg(input)
            flag = 1
            # pass
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        vis, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(vis)
        ir, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(ir)

        vis_2, vis_3, vis_4 = self.vgg19(vis)
        ir_2, ir_3, ir_4 = self.vgg19(ir)


        flag = 0

        x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
        conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
        x = self.max_pool1(conv1)

        x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
        conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
        x = self.max_pool2(conv2)

        x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
        conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
        x = self.max_pool3(conv3)

        x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))  # 256
        unet_out, unet_map = self.se_256(x)
        vgg_v_out, vgg_v_map = self.se_256(vis_4)
        vgg_i_out, vgg_i_map = self.se_256(ir_4)
        # maps.append(unet_map)
        # maps.append(vgg_v_map)
        # maps.append(vgg_i_map)
        # vis_c4 = torch.cat([unet_out, vgg_v_out], 1)  # 256 + 256
        # ir_c4 = torch.cat([unet_out, vgg_i_out], 1)  # 256 + 256
        att_4 = torch.cat([unet_out, vgg_v_out, vgg_i_out], 1)
        x = self.deconv4(att_4)  # 256*3 -> 256, deconv the concated attention maps
        conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))

        conv6 = F.upsample(conv4, scale_factor=2, mode='bilinear')  # 256
        # print(conv3.shape)
        # print(y_c_features[1].shape)
        # conv3 = conv3 * y_c_features[1] if self.opt.self_attention else conv3
        unet_out, unet_map = self.se_128(conv3) # conv3 128
        vgg_v_out, vgg_v_map = self.se_128(vis_3)
        vgg_i_out, vgg_i_map = self.se_128(ir_3)
        # maps.append(unet_map)
        # maps.append(vgg_v_map)
        # maps.append(vgg_i_map)
        # vis_c7 = torch.cat([unet_out, vgg_v_out], 1)  # 128(conv3) + 128
        # ir_c7 = torch.cat([unet_out, vgg_i_out], 1)  # 128 + 128
        att_7 = torch.cat([unet_out, vgg_v_out, vgg_i_out], 1)
        att_7 = self.att_deconv7(att_7) # 128*3 -> 128
        up7 = torch.cat([self.deconv6(conv6), att_7], 1) # deconv6, 256->128
        # up7 = self.deconv6(torch.cat([conv6, vis_c7, ir_c7], 1))  # 256 + 256 + 256 -> 256
        x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))  #
        conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))  # 128

        unet_out, unet_map = self.se_64(conv2)
        vgg_v_out, vgg_v_map = self.se_64(vis_2)
        vgg_i_out, vgg_i_map = self.se_64(ir_2)
        # maps.append(self.se_64(conv2))
        # maps.append(self.se_64(vis_2))
        # maps.append(self.se_64(ir_2))
        conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')
        # vis_c8 = torch.cat([unet_out, vgg_v_out], 1)  # 64(conv2) + 64
        # ir_c8 = torch.cat([unet_out, vgg_i_out], 1)  # 64 + 64
        att_8 = torch.cat([unet_out, vgg_v_out, vgg_i_out], 1)
        att_8 = self.att_deconv8(att_8) # 64*3 -> 64
        up8 = torch.cat([self.deconv7(conv7), att_8], 1) # deconv7, 128->64
        x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
        conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))  # 64

        conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')
        # conv1 = conv1 * gray if self.opt.self_attention else conv1
        # unet_out, unet_map = self.se_32(conv1)
        # vgg_v_out, vgg_v_map = self.se_32(vis_1)
        # vgg_i_out, vgg_i_map = self.se_32(ir_1)
        # maps.append(unet_out)
        # maps.append(conv1)
        # maps.append(vgg_i_out)
        # vis_c9 = torch.cat([unet_out, vgg_v_out], 1)  # 32 + 32
        # ir_c9 = torch.cat([unet_out, vgg_i_out], 1)  # 32 + 32
        # att_9 = torch.cat([unet_out, vgg_v_out, vgg_i_out], 1)
        # att_9 = self.att_deconv9(att_9) # 32*3 -> 32
        up9 = torch.cat([self.deconv8(conv8), conv1], 1) # deconv8, 64 -> 32
        x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
        conv9 = self.LReLU9_2(self.conv9_2(x))

        latent = self.conv10(conv9)

        latent = self.tanh(latent)
        # latent = (latent + 1) / 2
        output = latent

        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        # gray = pad_tensor_back(gray, pad_left, pad_right, pad_top, pad_bottom)
        if flag == 1:
            output = F.upsample(output, scale_factor=2, mode='bilinear')
        if self.skip:
            return output, latent
        else:
            return output


class Resnet_18(nn.Module):
    def __init__(self):
        super(Resnet_18, self).__init__()

        self.model = models.resnet18(pretrained=False)
        self.model.load_state_dict(torch.load('./models/resnet18-5c106cde.pth'))
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.to_pool_init = nn.Sequential()
        self.to_pool_64 = nn.Sequential()
        self.to_pool_128 = nn.Sequential()
        self.to_pool_256 = nn.Sequential()

        self.to_pool_init.add_module('0', self.model.conv1)
        self.to_pool_init.add_module('1', self.model.bn1)
        self.to_pool_init.add_module('2', self.model.relu)
        self.to_pool_init.add_module('3', self.model.maxpool)
        self.feature_1 = self.model.layer1
        counter = 4
        self.to_pool_64.add_module(str(counter), self.model.conv1)
        counter += 1
        self.to_pool_64.add_module(str(counter), self.model.bn1)
        counter += 1
        self.to_pool_64.add_module(str(counter), self.model.relu)
        counter += 1
        self.to_pool_64.add_module(str(counter), self.model.maxpool)
        counter += 1
        for i in range(2):
            self.to_pool_64.add_module(str(counter), self.feature_1[i].conv1)
            counter += 1
            self.to_pool_64.add_module(str(counter), self.feature_1[i].bn1)
            counter += 1
            self.to_pool_64.add_module(str(counter), self.feature_1[i].relu)
            counter += 1
            self.to_pool_64.add_module(str(counter), self.feature_1[i].conv2)
            counter += 1
            self.to_pool_64.add_module(str(counter), self.feature_1[i].bn2)
            counter += 1
        self.feature_2 = self.model.layer2
        for i in range(2):
            self.to_pool_128.add_module(str(counter), self.feature_2[i].conv1)
            counter += 1
            self.to_pool_128.add_module(str(counter), self.feature_2[i].bn1)
            counter += 1
            self.to_pool_128.add_module(str(counter), self.feature_2[i].relu)
            counter += 1
            self.to_pool_128.add_module(str(counter), self.feature_2[i].conv2)
            counter += 1
            self.to_pool_128.add_module(str(counter), self.feature_2[i].bn2)
            counter += 1
            if i == 0:
                self.to_pool_128.add_module(str(counter), self.feature_2[i].downsample[0])
                counter += 1
                self.to_pool_128.add_module(str(counter), self.feature_2[i].downsample[1])
                counter += 1
        self.feature_3 = self.model.layer3
        for i in range(2):
            self.to_pool_256.add_module(str(counter), self.feature_3[i].conv1)
            counter += 1
            self.to_pool_256.add_module(str(counter), self.feature_3[i].bn1)
            counter += 1
            self.to_pool_256.add_module(str(counter), self.feature_3[i].relu)
            counter += 1
            self.to_pool_256.add_module(str(counter), self.feature_3[i].conv2)
            counter += 1
            self.to_pool_256.add_module(str(counter), self.feature_3[i].bn2)
            counter += 1
            if i == 0:
                self.to_pool_256.add_module(str(counter), self.feature_3[i].downsample[0])
                counter += 1
                self.to_pool_256.add_module(str(counter), self.feature_3[i].downsample[1])
                counter += 1

        # print(self.to_pool_128)
        # print(self.to_pool_256)

    def forward(self, x):
        # print("FORWARD!")
        x_64 = self.to_pool_64(x)
        print(x_64.shape) # 10.64.16.16
        print(self.to_pool_128)
        x_128 = self.to_pool_128(x_64)

        x_256 = self.to_pool_256(x_128)
        return x_64, x_128, x_256




class Unet_res18(nn.Module):
    def __init__(self):
        super(Unet_res18, self).__init__()

        self.se_32 = SELayer(32)
        self.se_64 = SELayer(64)
        self.se_128 = SELayer(128)
        self.se_256 = SELayer(256)
        self.res18 = resnet18()
        for i, param in enumerate(self.res18.parameters()):
            param.requires_grad = False


        self.skip = False
        p = 1
        # self.conv1_1 = nn.Conv2d(4, 32, 3, padding=p)

        self.conv1_1 = nn.Conv2d(2, 32, 3, padding=p)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.max_pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.deconv4 = nn.Conv2d(256 * 4, 256, 3, padding=p)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.max_pool4 = nn.MaxPool2d(2)

        self.deconv6 = nn.Conv2d(768, 256, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_1 = nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_2 = nn.BatchNorm2d(128)

        # self.deconv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.deconv7 = nn.Conv2d(128 * 3, 128, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_1 = nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_2 = nn.BatchNorm2d(64)

        # self.deconv8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.deconv8 = nn.Conv2d(64 + 32, 64, 3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn9_1 = nn.BatchNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv10 = nn.Conv2d(32, 1, 1)

        self.tanh = nn.Tanh()

    def forward(self, ir, vis):
        maps = []
        # print(ir.shape)
        # print(vis.shape)
        input = torch.cat([ir, vis], 1)
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        vis, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(vis)
        ir, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(ir)

        vis_feature_64, vis_feature_128, vis_feature_256 = self.res18(vis)

        ir_feature_64, ir_feature_128, ir_feature_256 = self.res18(ir)

        if input.size()[3] > 2200:
            avg = nn.AvgPool2d(2)
            input = avg(input)
            flag = 1
            # pass
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        vis, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(vis)
        ir, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(ir)


        flag = 0

        x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
        conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
        x = self.max_pool1(conv1)
        conv1 = x # 0730, resnet

        x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
        conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
        x = self.max_pool2(conv2)
        conv2 = x # 0730, resnet

        x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
        conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
        x = self.max_pool3(conv3)
        conv3 = x # 0730, resnet

        x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))  # 256
        x = self.max_pool4(x)
        unet_out, unet_map = self.se_256(x)
        vgg_v_out, vgg_v_map = self.se_256(vis_feature_256)
        vgg_i_out, vgg_i_map = self.se_256(ir_feature_256)
        # maps.append(unet_map)
        # maps.append(vgg_v_map)
        # maps.append(vgg_i_map)
        vis_c4 = torch.cat([unet_out, vgg_v_out], 1)  # 256 + 256
        ir_c4 = torch.cat([unet_out, vgg_i_out], 1)  # 256 + 256
        x = self.deconv4(torch.cat([vis_c4, ir_c4], 1))  # 256*4 -> 256
        conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))

        conv6 = F.upsample(conv4, scale_factor=2, mode='bilinear')  # 256

        # conv3 = conv3 * y_c_features[1] if self.opt.self_attention else conv3
        unet_out, unet_map = self.se_128(conv3)
        vgg_v_out, vgg_v_map = self.se_128(vis_feature_128)
        vgg_i_out, vgg_i_map = self.se_128(ir_feature_128)
        # maps.append(unet_map)
        # maps.append(vgg_v_map)
        # maps.append(vgg_i_map)

        vis_c7 = torch.cat([unet_out, vgg_v_out], 1)  # 128(conv3) + 128
        ir_c7 = torch.cat([unet_out, vgg_i_out], 1)  # 128 + 128

        up7 = self.deconv6(torch.cat([conv6, vis_c7, ir_c7], 1))  # 256 + 256 + 256 -> 256
        x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))  #
        conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))  # 128

        unet_out, unet_map = self.se_64(conv2)
        vgg_v_out, vgg_v_map = self.se_64(vis_feature_64)
        vgg_i_out, vgg_i_map = self.se_64(ir_feature_64)
        # maps.append(unet_out)
        # maps.append(conv2)
        # maps.append(self.se_64(ir_2))
        conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')
        vis_c8 = torch.cat([unet_out, vgg_v_out], 1)  # 64(conv2) + 64
        ir_c8 = torch.cat([unet_out, vgg_i_out], 1)  # 64 + 64

        up8 = self.deconv7(torch.cat([conv7, vis_c8, ir_c8], 1))  # 128 + 128 + 128 -> 128
        x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
        conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))  # 64

        conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')
        # conv1 = conv1 * gray if self.opt.self_attention else conv1
        unet_out, unet_map = self.se_32(conv1)
        maps.append(unet_out)
        maps.append(conv1)
        # vgg_v_out, vgg_v_map = self.se_32(vis_1)
        # vgg_i_out, vgg_i_map = self.se_32(ir_1)
        # maps.append(unet_out)
        # maps.append(conv1)
        # maps.append(vgg_i_out)
        # vis_c9 = torch.cat([unet_out, vgg_v_out], 1)  # 32 + 32
        # ir_c9 = torch.cat([unet_out, vgg_i_out], 1)  # 32 + 32

        up9 = self.deconv8(torch.cat([conv8, unet_out], 1))  # 64 + 32-> 64
        x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
        conv9 = self.LReLU9_2(self.conv9_2(x))
        conv9 = F.upsample(conv9, scale_factor=2, mode='bilinear')

        latent = self.conv10(conv9)

        latent = self.tanh(latent)
        # latent = (latent + 1) / 2
        output = latent

        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        # gray = pad_tensor_back(gray, pad_left, pad_right, pad_top, pad_bottom)
        if flag == 1:
            output = F.upsample(output, scale_factor=2, mode='bilinear')
        if self.skip:
            return output, latent
        else:
            return output, maps

class Unet_2(nn.Module):
    def __init__(self):
        super(Unet_2, self).__init__()

        # self.vgg = load_vgg16("./core/Losses/vgg", cfg)
        # self.vgg.to_relu_1_2[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


        self.skip = False
        p = 1
        # self.conv1_1 = nn.Conv2d(4, 32, 3, padding=p)

        self.conv1_1 = nn.Conv2d(2, 32, 3, padding=p)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.max_pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.deconv4 = nn.Conv2d(256 * 4, 256, 3, padding=p)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.max_pool4 = nn.MaxPool2d(2)

        # self.conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
        # self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        # if self.opt.use_norm == 1:
        #     self.bn5_1 = SynBN2d(512) if self.opt.syn_norm else nn.BatchNorm2d(512)
        # self.conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        # self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        # if self.opt.use_norm == 1:
        #     self.bn5_2 = SynBN2d(512) if self.opt.syn_norm else nn.BatchNorm2d(512)

        # self.deconv5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        # self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        # self.conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
        # self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        # if self.opt.use_norm == 1:
        #     self.bn6_1 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)
        # self.conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
        # self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        # if self.opt.use_norm == 1:
        #     self.bn6_2 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)

        # self.deconv6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.deconv6 = nn.Conv2d(256+128, 256, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_1 = nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_2 = nn.BatchNorm2d(128)

        # self.deconv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.deconv7 = nn.Conv2d(128+64, 128, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_1 = nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_2 = nn.BatchNorm2d(64)

        # self.deconv8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.deconv8 = nn.Conv2d(64+32, 64, 3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn9_1 = nn.BatchNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv10 = nn.Conv2d(32, 1, 1)

        self.tanh = nn.Tanh()

    def forward(self, ir, vis):
        maps = []
        # print(ir.shape)
        # print(vis.shape)
        input = torch.cat([ir, vis], 1)
        if input.size()[3] > 2200:
            avg = nn.AvgPool2d(2)
            input = avg(input)
            flag = 1
            # pass
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        vis, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(vis)
        ir, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(ir)


        flag = 0

        x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
        conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
        x = self.max_pool1(conv1)

        x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
        conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
        x = self.max_pool2(conv2)

        x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
        conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
        x = self.max_pool3(conv3)

        x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))  # 256
        conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))

        conv6 = F.upsample(conv4, scale_factor=2, mode='bilinear')  # 256

        up7 = self.deconv6(torch.cat([conv6, conv3], 1))  # 256 + 128 -> 256
        x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))  #
        conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))  # 128

        conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')

        up8 = self.deconv7(torch.cat([conv7, conv2], 1))  # 128 + 64 -> 128
        x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
        conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))  # 64

        conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')

        up9 = self.deconv8(torch.cat([conv8, conv1], 1))  # 64 + 32 -> 64
        x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
        conv9 = self.LReLU9_2(self.conv9_2(x))

        latent = self.conv10(conv9)

        latent = self.tanh(latent)
        # latent = (latent + 1) / 2
        output = latent

        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        # gray = pad_tensor_back(gray, pad_left, pad_right, pad_top, pad_bottom)
        if flag == 1:
            output = F.upsample(output, scale_factor=2, mode='bilinear')
        if self.skip:
            return output, latent
        else:
            return output



from utils import *

# CHANNEL_DIM = 1
# hidden_dim = [16, 32, 64, 128]
n = 32
l = 0.2


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # self.conv0 = nn.Conv2d(2, int(n/2), 1, padding=0)
        self.conv1 = nn.Conv2d(2, n, 3, padding=1)
        self.conv2 = nn.Conv2d(n, n, 3, padding=1)
        self.conv3 = nn.Conv2d(n * 2, n, 3, padding=1)
        self.conv4 = nn.Conv2d(n * 3, n, 3, padding=1)

        self.l1 = nn.LeakyReLU(l)

        self.n1 = nn.BatchNorm2d(n)
        self.n2 = nn.BatchNorm2d(n)
        self.n3 = nn.BatchNorm2d(n)
        self.n4 = nn.BatchNorm2d(n)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.l1(self.n1(x1))  # +x1

        x2 = self.conv2(x1)
        x2 = self.l1(self.n2(x2))  # + x2

        x3 = self.conv3(torch.cat((x1, x2), 1))
        x3 = self.l1(self.n3(x3))

        x4 = self.conv4(torch.cat((x1, x2, x3), 1))
        x4 = self.l1(self.n4(x4))

        output = torch.cat((x1, x2, x3, x4), 1)

        return output


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv5 = nn.Conv2d(n * 4, n * 4, 3, padding=1)
        self.conv6 = nn.Conv2d(n * 4, n * 2, 3, padding=1)
        self.conv7 = nn.Conv2d(n * 2, n, 3, padding=1)
        self.conv8 = nn.Conv2d(n, 1, 1, padding=0)
        self.l1 = nn.LeakyReLU(l)

        self.n1 = nn.BatchNorm2d(n * 4)
        self.n2 = nn.BatchNorm2d(n * 2)
        self.n3 = nn.BatchNorm2d(n)

    def forward(self, x):
        x = self.conv5(x)
        x = self.l1(self.n1(x))

        x = self.conv6(x)
        x = self.l1(self.n2(x))

        x = self.conv7(x)
        x = self.l1(self.n3(x))

        x = torch.tanh(self.conv8(x))
        return x


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output

if __name__ == '__main__':
    model = Resnet_18()







