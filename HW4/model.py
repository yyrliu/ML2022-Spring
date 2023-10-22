import torch
from torch import nn
from torchinfo import summary

from comformer import Comformer
from loss_fn import AMSoftmax

class Classifier(nn.Module):
    def __init__(self, *, input_mels, d_model, n_class, net_conf, loss_fn_conf):
        super().__init__()

        self.input_mels = input_mels

        self.comformer = Comformer(
            input_mels=input_mels,
            d_model=d_model,
            post_comformer_dropout=net_conf["post_comformer_dropout"],
            pred_layer=net_conf["pred_layer"],
            comformer_conf=net_conf["comformer_conf"],
        )
        self.classifier = AMSoftmax(
            in_feats=d_model,
            n_class=n_class,
            **loss_fn_conf
        )

    def forward(self, mels, lb):
        stats = self.comformer(mels)
        costh, loss = self.classifier(stats, lb)
        return costh, loss
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def summerize(self, len=128):
        x = torch.randn((1, len, self.input_mels)).to(self.device)
        lb = torch.zeros((1,)).long().to(self.device)
        summary(
            self,
            input_data=[x, lb],
            device=self.device,
            depth=7
        )

    
if __name__ == "__main__":

    conf = {
        "input_mels": 40,
        "d_model": 80,
        "n_class": 600,
        "loss_fn_conf": {
            "m": 0.2,
            "s": 30,
            "norm_affine": True,
            "feat_norm": True
        },
        "net_conf": {
            "pred_layer": 0,
            "post_comformer_dropout": 0.1,
            "comformer_conf":{
                "layers": 2,
                "norm_after_cf_block": False,
                "submodules":{
                    "transformer":{
                        "nhead": 2,
                        "dim_feedforward": 265
                    },
                    "conv": {
                        "version": 2,
                        "kernel_size": 31,
                        "dropout": 0.1,
                        "module_factor": 1.0,
                        "input_factor": 1.0
                    },
                    "feedforward": {
                        "version": 1,
                        "dropout": 0.1,
                        "module_factor": 0.5,
                        "input_factor": 1.0
                    }
                }
            }
        }
    }

    Classifier(**conf).summerize()
