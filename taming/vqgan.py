import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from taming.diffusion import Encoder, Decoder
from taming.quantize import VectorQuantizer2 as VectorQuantizer


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 n_embed,   # vocab_size
                 embed_dim,   # dim of token vector
                 ckpt_path=None,
                 ignore_keys=[],
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,   # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        # self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        sd_self = self.state_dict()
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
            if k in sd_self and sd_self[k].shape != sd[k].shape:
                print(f"Deleting key {k} from state_dict.")
                del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        # x: [B, 3, H, W]
        h = self.encoder(x)
        # [B, C_z, h, w]

        h = self.quant_conv(h)
        # [B, C_embed, h, w]

        quant, emb_loss, info = self.quantize(h)
        # quant: [B, C_embed, h, w]
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        # [B, C_z, h, w]

        dec = self.decoder(quant)
        # [B, 3, H, W]
        return dec

    def decode_code(self, code_b, shape):
        quant_b = self.quantize.get_codebook_entry(code_b, shape)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch):
        x = batch
        if len(x.shape) == 3:
            x = x[..., None]
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log
