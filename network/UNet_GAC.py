import torch.nn
import torch.nn.functional as F

from .unet_parts import *

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv2d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet_GAC(nn.Module):
    def __init__(self, n_channels, n_classes, normalization='none', bilinear=True):
        super(UNet_GAC, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.block_two_up = UpsamplingDeconvBlock(128, 64, normalization=normalization)

        self.block_three_up1 = UpsamplingDeconvBlock(256, 128, normalization=normalization)
        self.block_three_up2 = UpsamplingDeconvBlock(128, 64, normalization=normalization)
        self.block_four_up1 = UpsamplingDeconvBlock(512, 256, normalization=normalization)
        self.block_four_up2 = UpsamplingDeconvBlock(256, 128, normalization=normalization)
        self.block_four_up3 = UpsamplingDeconvBlock(128, 64, normalization=normalization)
        # self.block_b = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.block_boundary = nn.Conv2d(64*3, n_classes, kernel_size=1, stride=1, padding=0)

        self.dropout = nn.Dropout2d(p=0.5, inplace=False)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        out_dis = self.tanh(out)
        return  out_dis
    def boundary_decoder(self,features):
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x2_up = self.block_two_up(x2)
        x3_1 = self.block_three_up1(x3)
        x3_2 = self.block_three_up2(x3_1)
        x4_1 = self.block_four_up1(x4)
        x4_2 = self.block_four_up2(x4_1)
        x4_3 = self.block_four_up3(x4_2)
        x = torch.cat([x2_up,x3_2,x4_3],dim = 1)
        x = self.block_boundary(x)
        out = self.sig(x)
        return out


    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out_dis = self.decoder(features)
        boundary = self.boundary_decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return  out_dis,boundary

