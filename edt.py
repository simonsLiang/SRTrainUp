import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from easydict import EasyDict as edict
from . import restormer


class ResBlockDown(nn.Module):
    def __init__(self, in_chl, out_chl, down=False):
        super(ResBlockDown, self).__init__()
        self.in_chl = in_chl
        self.out_chl = out_chl

        self.conv_1 = nn.Conv2d(in_chl, in_chl, 3, 1, 1, bias=True)
        self.relu_1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv_2 = nn.Conv2d(in_chl, out_chl, 3, 1, 1, bias=True)
        self.relu_2 = nn.LeakyReLU(0.2, inplace=False)
        self.shortcut = nn.Conv2d(in_chl, out_chl, 1, 1, 0, bias=True)

        self.down = down
        if down:
            self.conv_down = nn.Conv2d(out_chl, out_chl, 4, 2, 1, bias=False)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.relu_1(self.conv_1(x))
        x = self.relu_2(self.conv_2(x))
        x += identity

        if self.down:
            x_down = self.conv_down(x)
            return x_down, x
        else:
            return x


class ResBlockUp(nn.Module):
    def __init__(self, in_chl, out_chl):
        super(ResBlockUp, self).__init__()
        self.in_chl = in_chl
        self.out_chl = out_chl

        self.conv_1 = nn.Conv2d(in_chl, out_chl, 3, 1, 1, bias=True)
        self.relu_1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv_2 = nn.Conv2d(out_chl, out_chl, 3, 1, 1, bias=True)
        self.relu_2 = nn.LeakyReLU(0.2, inplace=False)
        self.shortcut = nn.Conv2d(in_chl, out_chl, 1, 1, 0, bias=True)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.relu_1(self.conv_1(x))
        x = self.relu_2(self.conv_2(x))
        x += identity

        return x

class UpResBlock(nn.Module):
    def __init__(self, in_chl, out_chl):
        super(UpResBlock, self).__init__()
        self.in_chl = in_chl
        self.out_chl = out_chl

        self.up = nn.ConvTranspose2d(in_chl, out_chl, kernel_size=2, stride=2, bias=True)
        self.block = ResBlockUp(out_chl * 2, out_chl)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)

        return x


class ResBlockSkip(nn.Module):
    def __init__(self, in_chl, out_chl):
        super(ResBlockSkip, self).__init__()
        self.in_chl = in_chl
        self.out_chl = out_chl

        self.conv = nn.Conv2d(in_chl, out_chl, 3, 1, 1, bias=True)
        self.block = ResBlockUp(out_chl * 2, out_chl)

    def forward(self, x, skip):
        x = self.conv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)

        return x


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, input_resolution=None):
        self.input_resolution = input_resolution
        self.scale = scale
        self.num_feat = num_feat
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        if (self.scale & (self.scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(self.scale, 2))):
                flops += H * W * self.num_feat * self.num_feat * 4 * 9
        elif self.scale == 3:
            flops += H * W * self.num_feat * self.num_feat * 9 * 9
        return flops


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        self.num_out_ch = num_out_ch
        self.scale = scale
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)



class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        num_feat = config.MODEL.NUM_FEAT
        img_chl = config.MODEL.IN_CHANNEL
        embed_dim = config.MODEL.EMBED_DIM
        depth = config.MODEL.DEPTH

        self.num_feat = num_feat
        self.embed_dim = embed_dim
        self.depth = depth

        self.scales = config.MODEL.SCALES

        # preprocessing / postprocessing
        self.img_range = config.MODEL.IMAGE_RANGE
        if img_chl == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        # heads
        for s in self.scales:
            head = nn.ModuleList()
            head.append(nn.Conv2d(img_chl, num_feat, 3, 1, 1))
            for i in range(depth):
                head.append(ResBlockDown(num_feat*2**i, num_feat*2**(i+1), down=False))
            head.append(nn.Conv2d(num_feat*2**depth, embed_dim, 3, 1, 1))
            setattr(self, 'head_sr_x%d' % s, head)
        #body
        self.body = nn.Sequential(
            restormer.TB(dim=60,num_heads=1),
            restormer.TB(dim=60,num_heads=1),
            restormer.TB(dim=60,num_heads=2),
            restormer.TB(dim=60,num_heads=2),
            restormer.TB(dim=60,num_heads=3),
            restormer.TB(dim=60,num_heads=3))

        # tails
        for s in self.scales:
            tail = nn.ModuleList()
            for i in reversed(range(depth)):
                in_chl = embed_dim if i == depth - 1 else num_feat * 2 ** (i + 2)
                out_chl = num_feat * 2 ** (i + 1)
                tail.append(ResBlockSkip(in_chl, out_chl))
            if config.MODEL.UPSAMPLER == 'pixelshuffle':
                tail.append(Upsample(s, out_chl))
                tail.append(nn.Conv2d(out_chl, img_chl, 3, 1, 1))
            else:
                tail.append(UpsampleOneStep(s, out_chl, img_chl))
                tail.append(nn.Identity())
            setattr(self, 'tail_sr_x%d' % s, tail)


    def forward(self, lqs, gt=None):
        # preprocessing
        self.mean = self.mean.type_as(lqs[0])
        lqs = [(lq - self.mean) * self.img_range for lq in lqs]
        n_sr = len(self.scales)

        # head
        skips_all = []
        outs = []
        ### sr
        for i, s in enumerate(self.scales):
            skips = []
            x = lqs[i].clone()
            head = getattr(self, 'head_sr_x%d' % s)
            for j, block in enumerate(head):
                x = block(x)
                if 0 < j < len(head) - 1:
                    skips.append(x)
            skips_all.append(skips)
            outs.append(x)
       
        
        x_b = [self.body(x) for x in outs]

        # tail
        outs = []
        ### sr
        for i, s in enumerate(self.scales):
            x = x_b[i]
            tail = getattr(self, 'tail_sr_x%d' % s)
            for j, block in enumerate(tail):
                if j == len(tail) - 1:
                    lq_up = F.interpolate(lqs[i], scale_factor=s, mode='bilinear', align_corners=False)
                    x = lq_up + block(x)
                elif j == len(tail) - 2:
                    x = block(x)
                else:
                    x = block(x, skips_all[i][-j-1])
            outs.append(x)

        # preprocessing
        preds = [x / self.img_range + self.mean for x in outs]
        return preds

class Config:
    MODEL = edict()
    MODEL.IN_CHANNEL = 3
    MODEL.DEPTH = 2
    MODEL.SCALES = [4]
    MODEL.IMAGE_RANGE = 1.0
    MODEL.NUM_FEAT = 8
    MODEL.EMBED_DIM = 60
    MODEL.UPSAMPLER = 'pixelshuffledirect'

class min_edt(nn.Module):
    def __init__(self):
        super(min_edt, self).__init__()
        self.config = Config()
        self.network = Network(self.config)
    def forward(self, x):
        out = self.network(x)
        return out[0]


