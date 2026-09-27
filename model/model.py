from torch import Tensor
from torch import nn
from typing import Optional


from .lraspp import LRASPP
from .decoder import Projection
from .convresnet import ConvResEncoder
from .conresdecoder import ConResRecurrentDecoder

class MattingNetwork(nn.Module):
    def __init__(self,
                 variant: str = 'ConvResNet',
                 classes = 1,
                 pretrained_backbone: bool = False):
        super().__init__()
        assert variant == 'ConvResNet'
        self.variant = variant
        
        self.backbone = ConvResEncoder()
        self.aspp = LRASPP(768, 768)
        self.decoder = ConResRecurrentDecoder([96, 192, 384, 768], [512, 256, 128, 64])
        self.project_seg = Projection(64, classes)

        
    def forward(self,
                src: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None):
        

        src_sm = src
        
        f1, f2, f3, f4 = self.backbone(src_sm)


        f4 = self.aspp(f4)

        
        hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)

        seg = self.project_seg(hid.contiguous())
        return [seg, *rec]

