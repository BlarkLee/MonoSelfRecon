# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *


def conv(in_planes, out_planes, kernel_size, instancenorm=False):
    if instancenorm:
        m = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.InstanceNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        m = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    return m


class DepthDecoder(nn.Module):
    def tuple_to_str(self, key_tuple):
        key_str = '-'.join(str(key_tuple))
        return key_str

    def __init__(self, num_ch_enc, embedder, embedder_out_dim,
                 use_alpha=False, scales=range(4), num_output_channels=4,
                 use_skips=True, sigma_dropout_rate=0.0, **kwargs):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.use_alpha = use_alpha
        self.sigma_dropout_rate = sigma_dropout_rate

        self.embedder = embedder
        self.E = embedder_out_dim

        final_enc_out_channels = num_ch_enc[-1]
        self.downsample = nn.MaxPool2d(3, stride=2, padding=1)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv_down1 = conv(final_enc_out_channels, 512, 1, False)
        self.conv_down2 = conv(512, 256, 3, False)
        self.conv_up1 = conv(256, 256, 3, False)
        self.conv_up2 = conv(256, final_enc_out_channels, 1, False)
        #self.conv_down1 = conv(final_enc_out_channels, 128, 1, False)
        #self.conv_down2 = conv(128, 64, 3, False)
        #self.conv_up1 = conv(64, 64, 3, False)
        #self.conv_up2 = conv(64, final_enc_out_channels, 1, False)
        #self.conv_down1 = conv(final_enc_out_channels, 64, 1, False)
        #self.conv_down2 = conv(64, 32, 3, False)
        #self.conv_up1 = conv(32, 32, 3, False)
        #self.conv_up2 = conv(32, final_enc_out_channels, 1, False)

        self.num_ch_enc = num_ch_enc
        #print("num_ch_enc=", num_ch_enc)
        self.num_ch_enc = [x + self.E for x in self.num_ch_enc]
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        #self.num_ch_dec = np.array([16, 32, 64])
        # self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        # decoder
        self.convs = nn.ModuleDict()
        for i in range(2, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[self.tuple_to_str(("upconv", i, 0))] = ConvBlock(num_ch_in, num_ch_out)
            #print("upconv_{}_{}".format(i, 0), num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[self.tuple_to_str(("upconv", i, 1))] = ConvBlock(num_ch_in, num_ch_out)
            #print("upconv_{}_{}".format(i, 1), num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[self.tuple_to_str(("dispconv", s))] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)


        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, disparity):
        input_features = list(input_features)
        B, S = disparity.size()
        #print("disparity", disparity.shape)
        disparity = self.embedder(disparity.reshape(B * S, 1)).unsqueeze(2).unsqueeze(3)
        #print("disparity", disparity.shape)

        # extension of encoder to increase receptive field
        input_features = input_features[:3]
        '''
        for i in input_features:
            print("input_features", i.shape)'''

        encoder_out = input_features[-1]
        #print("encoder_out", encoder_out.shape)
        conv_down1 = self.conv_down1(self.downsample(encoder_out))
        #print("conv_down1", conv_down1.shape)
        conv_down2 = self.conv_down2(self.downsample(conv_down1))
        #print("conv_down2", conv_down2.shape)
        conv_up1 = self.conv_up1(self.upsample(conv_down2))
        #print("conv_up1", conv_up1.shape)
        conv_up2 = self.conv_up2(self.upsample(conv_up1))
        #print("conv_up2", conv_up2.shape)
        conv_up2 = F.interpolate(conv_up2, size=(encoder_out.shape[-2], encoder_out.shape[-1]), mode="nearest")
        #print("conv_up2", conv_up2.shape)

        # repeat / reshape features
        _, C_feat, H_feat, W_feat = conv_up2.size()
        feat_tmp = conv_up2.unsqueeze(1).expand(B, S, C_feat, H_feat, W_feat) \
            .contiguous().view(B * S, C_feat, H_feat, W_feat)
        disparity_BsCHW = disparity.repeat(1, 1, H_feat, W_feat)
        #print("disparity_BsCHW", disparity_BsCHW.shape)
        conv_up2 = torch.cat((feat_tmp, disparity_BsCHW), dim=1)
        #print("conv_up2", conv_up2.shape)
        
        # repeat / reshape features
        for i, feat in enumerate(input_features):
            _, C_feat, H_feat, W_feat = feat.size()
            feat_tmp = feat.unsqueeze(1).expand(B, S, C_feat, H_feat, W_feat) \
                .contiguous().view(B * S, C_feat, H_feat, W_feat)
            disparity_BsCHW = disparity.repeat(1, 1, H_feat, W_feat)
            #print("feat_tmp", feat_tmp.shape)
            #print("disparity_BsCHW", disparity_BsCHW.shape)
            input_features[i] = torch.cat((feat_tmp, disparity_BsCHW), dim=1)
            #print("input_features[i]", input_features[i].shape)
        
        # for i, feat in enumerate(input_features):
        #     _, C_feat, H_feat, W_feat = feat.size()
        #     input_features[i] = feat.unsqueeze(1).expand(B, S, C_feat, H_feat, W_feat) \
        #         .contiguous().view(B * S, C_feat, H_feat, W_feat)

        # decoder
        outputs = {}
        # x = input_features[-1]
        x = conv_up2
        for i in range(2, -1, -1):
            x = self.convs[self.tuple_to_str(("upconv", i, 0))](x)
            #print("x", x.shape)
            x = [upsample(x)]
            #print("x[0]", x[0].shape)
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            #print("x", x.shape)
            x = self.convs[self.tuple_to_str(("upconv", i, 1))](x)
            #print("x", x.shape)
            if i in self.scales:
                output = self.convs[self.tuple_to_str(("dispconv", i))](x)
                #print("output", output.shape)
                H_mpi, W_mpi = output.size(2), output.size(3)
                mpi = output.view(B, S, 4, H_mpi, W_mpi)
                #print("mpi", mpi.shape)
                mpi_rgb = self.sigmoid(mpi[:, :, 0:3, :, :])
                #print("mpi_rgb", mpi_rgb.shape)
                mpi_sigma = torch.abs(mpi[:, :, 3:, :, :]) + 1e-4 \
                        if not self.use_alpha \
                        else self.sigmoid(mpi[:, :, 3:, :, :])
                #print("mpi_sigma", mpi_sigma.shape)

                if self.sigma_dropout_rate > 0.0 and self.training:
                    mpi_sigma = F.dropout2d(mpi_sigma, p=self.sigma_dropout_rate)
                    
                outputs[("disp", i)] = torch.cat((mpi_rgb, mpi_sigma), dim=2)
                #print("outputs[('disp', i)]", i, outputs[("disp", i)].shape)
                #print("outputs[(disp, i)]", outputs[("disp", i)].shape)
        #stop

        return outputs
