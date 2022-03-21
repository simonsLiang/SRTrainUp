import torch.nn as nn
from . import block as B
import torch
from . import  swinir

class decoder2(nn.Module):
    def __init__(self, in_nc=48, nf=48, out_nc=3, upscale=4):
        super(decoder2, self).__init__()
        self.de = nn.Sequential(
        B.conv_layer(in_channels=in_nc,
                    out_channels=nf, kernel_size=3),
        nn.PReLU(),
        B.pixelshuffle_block(in_channels=nf,
                    out_channels=nf, upscale_factor=2),
        nn.PReLU(),
        B.conv_layer(in_channels=nf,
                    out_channels=nf, kernel_size=5),
        nn.PReLU(),
        B.pixelshuffle_block(in_channels=nf,
                    out_channels=out_nc, upscale_factor=2),
    )

    def forward(self, input):

        out_fea = self.de(input)

        return out_fea
        
class decoder(nn.Module):
    def __init__(self, in_nc=48, nf=48, out_nc=3, upscale=4):
        super(decoder, self).__init__()
        self.de = nn.Sequential(
        B.conv_layer(in_channels=in_nc,
                    out_channels=nf, kernel_size=3),
        nn.PReLU(),
        B.pixelshuffle_block(in_channels=nf,
                    out_channels=nf//4, upscale_factor=2),
        nn.PReLU(),
        B.conv_layer(in_channels=nf//4,
                    out_channels=nf//4, kernel_size=5),
        nn.PReLU(),
        B.pixelshuffle_block(in_channels=nf//4,
                    out_channels=out_nc, upscale_factor=2,kernel_size=5),
    )

    def forward(self, input):

        out_fea = self.de(input)

        return out_fea
  
class encoder(nn.Module):
    def __init__(self, in_nc=3, out_nc=48):
        super(encoder, self).__init__()
        self.en = nn.Sequential(
        B.conv_layer(in_channels=in_nc,
                    out_channels=in_nc*4, kernel_size=5,stride=2),
        nn.PReLU(),
        B.conv_layer(in_channels=in_nc*4,
                    out_channels=in_nc*4, kernel_size=5),
        nn.PReLU(),
        B.conv_layer(in_channels=in_nc*4,
                    out_channels=out_nc, kernel_size=3, stride=2),
        nn.PReLU(),
        B.conv_layer(in_channels=out_nc,
                    out_channels=out_nc, kernel_size=3),
    )
        
    def forward(self, input):
        output = self.en(input)
        return output

class teacher(nn.Module):
    def __init__(self):
        super(teacher, self).__init__()
        self.en = encoder()
        self.de = decoder()
    def forward(self, input):
        output = self.en(input)
        output = self.de(output)
        return output
    def freeze(self):
        for param in self.en.parameters():
          param.requires_grad = False
        for param in self.de.parameters():
          param.requires_grad = False

class student(nn.Module):
    def __init__(self, in_nc=3, nf=48, out_nc=3, upscale=4):
        super(student, self).__init__()
        self.head = B.conv_layer(in_channels=in_nc,
                    out_channels=nf, kernel_size=3)
        self.en = nn.Sequential(
        B.conv_layer(in_channels=nf,
                    out_channels=nf, kernel_size=3),
        nn.PReLU(),
        B.conv_layer(in_channels=nf,
                    out_channels=nf, kernel_size=5),
        nn.PReLU(),
        B.conv_layer(in_channels=nf,
                    out_channels=nf, kernel_size=3),
    )
        self.pe = swinir.PatchEmbed(img_size=48,embed_dim=48)
        self.pue = swinir.PatchUnEmbed(img_size=48,embed_dim=48)
        self.transformer = swinir.SwinTransformerBlock(dim=48, input_resolution=(48,48), num_heads=6, window_size=8,mlp_ratio=2)

        self.de = decoder()
        for param in self.de.parameters():
            param.requires_grad = False

    def forward(self, input):
        _,_,H,W = input.shape
        out_fea = self.head(input)
        out_fea = self.en(out_fea)
        out_fea = self.pue(self.transformer(self.pe(out_fea),(H,W)),(H,W))
        output = self.de(out_fea)
        return out_fea,output
    
    def load_decode_dict(self,state):
      self.de.load_state_dict(state)

    def freeze_decode(self):
      for param in self.de.parameters():
        param.requires_grad = False
