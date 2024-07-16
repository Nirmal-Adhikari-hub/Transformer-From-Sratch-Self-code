import torch
import torch.nn as nn
from architecture import EncoderBlock, DecoderBlock




class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderBlock()
        self.decoder = DecoderBlock()

    def forward(self, src, tgt, src_pad_mask=None, tgt_pad_mask=None):
        # print(src.shape, tgt.shape, src_pad_mask.shape, tgt_pad_mask.shape)
        encoder_out = self.encoder(src, pad_mask=src_pad_mask)
        out = self.decoder(tgt, encoder_out=encoder_out, pad_mask=tgt_pad_mask)
        return out