import torch
from torch import nn
from torchinfo import summary

from .modules import ComformerBlock, SelfAttentionPooling

import logging
logger = logging.getLogger(__name__)

class PreNet(nn.Module):
    def __init__(self, input_mels, d_model, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.prenet = nn.Linear(input_mels, d_model)
        logger.info(f"PreNet: {input_mels} -> {d_model}, dropout={dropout}")
    
    def forward(self, mels):
        mels = self.prenet(mels)
        return nn.functional.dropout(mels, p=self.dropout)

class Encoder(nn.Module):
    def __init__(self, d_model, *, type, layers, **kargs):
        super().__init__()
        if type == "transformer":
            self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(
                d_model=d_model,
                batch_first=True,
                **kargs["submodules"]["encoder_layer"]
            ) for _ in range(layers)])
            
        elif type == "comformer":
            self.encoder = nn.ModuleList([ComformerBlock(
                d_model=d_model,
                **kargs["submodules"]
            ) for _ in range(layers)])

        else:
            raise NotImplementedError(f"Encoder type `{type}` not implemented.")

        logger.info(f"Encoder: {type} x {layers}")
        
    def forward(self, mels):
        for layer in self.encoder:
            mels = layer(mels)
        return mels

class Pooling(nn.Module):
    def __init__(self, d_model, *, type, **kargs):
        super().__init__()
        self.type = type
        
        if type == "self_attention":
            self.pooling = SelfAttentionPooling(d_model, **kargs)
        elif type == "mean" or type == "max":
            pass
        else:
            raise NotImplementedError(f"Pooling type `{type}` not implemented.")
        
        logger.info(f"Pooling: {type}")
        
    def forward(self, mels):
        if self.type == "mean":
            out = mels.mean(dim=1)
        elif self.type == "max":
            out = mels.max(dim=1)
        elif self.type == "self_attention":
            out = self.pooling(mels)

        return out
        
class Classifier(nn.Module):
    def __init__(self, d_model, n_class, num_layers=0):
        """
        args:
            num_layers: number of layers of prediction layers.
            if num_layers == 0, the classifier is a single linear layer without activation.
        """
        super().__init__()
        self.pred_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU()
            ) for _ in range(num_layers)]
        )
        self.pred_layers.append(nn.Linear(d_model, n_class))

    def forward(self, stats):
        for pred_layer in self.pred_layers:
            stats = pred_layer(stats)
        return stats

class Model(nn.Module):
    def __init__(self, input_mels, prenet, encoder, pooling, classifier):
        super().__init__()
        self.input_mels = input_mels
        self.prenet = prenet
        self.encoder = encoder
        self.pooling = pooling
        self.classifier = classifier

    def forward(self, mels):
        out = self.prenet(mels)
        out = self.encoder(out)
        out = self.pooling(out)
        out = self.classifier(out)
        return out
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def summerize(self, depth=7, batch=4, len=128):
        summary(
            self,
            input_size=[(batch, len, self.input_mels)],
            device=self.device,
            depth=depth
        )

def make_model(*, input_mels, d_model, n_class, conf):
    """
    Model is constisted of 4 parts:
        1. Prenet: project the dimension of features from that of input into d_model.
        2. Encoder: a stack of transformer or comformer layers.
        3. Pooling: pool the output of encoder into a fixed dimension.
        4. Classifier: project the the dimension of features from d_model into speaker nums.
    args:
        input_mels: the dimension of input features.
        d_model: the dimension of model.
        n_class: the number of speakers.
        conf: the configuration of model.
    """
    logger.info(f"Buliding model with input_mels={input_mels}, d_model={d_model}, n_class={n_class}")
    logger.info(f"Model conf: {conf}")

    prenet = PreNet(input_mels, d_model)
    encoder = Encoder(d_model, **conf["encoder"])
    pooling = Pooling(d_model, **conf["pooling"])
    classifier = Classifier(d_model, n_class)

    return Model(input_mels, prenet, encoder, pooling, classifier)
    
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    from model_conf import model_default_conf, comformer_default_conf

    model = make_model(**comformer_default_conf).summerize(depth=3)
