import torch.nn as nn
from utils import Reshape


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bn=True, a_func='lrelu'):

            block = nn.ModuleList()
            block.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            if bn:
                block.append(nn.BatchNorm2d(out_channels))
            if a_func == 'lrelu':
                block.append(nn.LeakyReLU(0.2))
            elif a_func == 'relu':
                block.append(nn.ReLU())
            else:
                pass

            return block

        def convTranspose_block(in_channels, out_channels, kernel_size, stride=2,
                 padding=0, output_padding=0, bn=True, a_func='relu'):
            '''
            H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding
            :param in_channels:
            :param out_channels:
            :param kernel_size:
            :param stride:
            :param padding:
            :param output_padding:
            :param bn:
            :param a_func:
            :return:
            '''
            block = nn.ModuleList()
            block.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                 padding, output_padding))
            if bn:
                block.append(nn.BatchNorm2d(out_channels))
            if a_func == 'lrelu':
                block.append(nn.LeakyReLU(0.2))
            elif a_func == 'relu':
                block.append(nn.ReLU())
            else:
                pass

            return block


        def encoder():
            conv_layer = nn.ModuleList()
            conv_layer += conv_block(3, 128, 5, 2, 2, False)    # 32x32x128
            conv_layer += conv_block(128, 256, 5, 2, 2)        # 16x16x256
            conv_layer += conv_block(256, 512, 5, 2, 2)         # 8x8x512
            conv_layer += conv_block(512, 1024, 5, 2, 2)       # 4x4x1024
            conv_layer += conv_block(1024, 64, 4, 1)          # 1x1x64
            return conv_layer

        def decoder():
            conv_layer = nn.ModuleList()
            conv_layer += conv_block(64, 4 * 4 * 1024, 1, a_func='relu')
            conv_layer.append(Reshape((1024, 4, 4)))                            # 4x4x1024
            conv_layer += convTranspose_block(1024, 512, 4, 2, 1)               # 8x8x512
            conv_layer += convTranspose_block(512, 256, 4, 2, 1)                # 16x16x256
            conv_layer += convTranspose_block(256, 128, 4, 2, 1)                # 32x32x128
            conv_layer += convTranspose_block(128, 3, 4, 2, 1, bn=False, a_func='')     # 64x64x3
            conv_layer.append(nn.Tanh())
            return conv_layer

        self.net = nn.Sequential(
            *encoder(),
            *decoder(),
        )

    def forward(self, input):
        out = self.net(input)
        return out

class DiscriminatorR(nn.Module):
    def __init__(self):
        super(DiscriminatorR, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size, stride=1,
                       padding=0, bn=True, a_func=True):

            block = nn.ModuleList()
            block.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            if bn:
                block.append(nn.BatchNorm2d(out_channels))
            if a_func:
                block.append(nn.LeakyReLU(0.2))

            return block


        self.net = nn.Sequential(
            *conv_block(3, 128, 5, 2, 2, False),                            # 32x32x128
            *conv_block(128, 256, 5, 2, 2),                                 # 16x16x256
            *conv_block(256, 512, 5, 2, 2),                                 # 8x8x512
            *conv_block(512, 1024, 5, 2, 2),                                # 4x4x1024
            *conv_block(1024, 1, 4, bn=False, a_func=False),                # 1x1x1
            nn.Sigmoid(),
        )

    def forward(self, img):
        out = self.net(img)
        return out

class DiscriminatorA(nn.Module):
    def __init__(self):
        super(DiscriminatorA, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size, stride=1,
                       padding=0, bn=True, a_func=True):

            block = nn.ModuleList()
            block.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            if bn:
                block.append(nn.BatchNorm2d(out_channels))
            if a_func:
                block.append(nn.LeakyReLU(0.2))

            return block

        self.net = nn.Sequential(
            *conv_block(6, 128, 5, 2, 2, False),                # 32x32x128
            *conv_block(128, 256, 5, 2, 2),                     # 16x16x256
            *conv_block(256, 512, 5, 2, 2),                     # 8x8x512
            *conv_block(512, 1024, 5, 2, 2),                    # 4x4x1024
            *conv_block(1024, 1, 4, bn=False, a_func=False),    # 1x1x1
            nn.Sigmoid(),
        )

    def forward(self, img):
        out = self.net(img)
        return out