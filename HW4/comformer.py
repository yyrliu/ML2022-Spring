import torch
import torch.nn as nn
from torchinfo import summary

class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)

class ResidualConnectionModule(nn.Module):
    """
    Residual Connection Module.
    outputs = (module(inputs) x module_factor + inputs x input_factor)
    """
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0):
        super().__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs):
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)
    
class FeedForwardModule(nn.Module):
    """
    Feed Forward Module.
    """
    def __init__(self, d_model, version: int = 2, dropout: float = 0.1):
        super().__init__()

        if version == 1:
            self.sequential = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.Dropout(dropout),
            )

        elif version == 2:
            self.sequential = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            
        else:
            raise ValueError("Version should be 1 or 2.")

    def forward(self, inputs):
        return self.sequential(inputs)
    
class ConvolutionModule(nn.Module):
    """
    Convolution Module.
    """
    def __init__(self, d_model, version: int = 2, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        if version == 1:
            self.sequential = nn.Sequential(
                nn.LayerNorm(d_model),          # (batch, length, d_model) -> (batch, length, d_model)
                Transpose(shape=(1, 2)),        # (batch, length, d_model) -> (batch, d_model, length)
                nn.Conv1d(d_model, d_model, kernel_size, padding=(kernel_size - 1) // 2),
                # (batch, d_model, length) -> (batch, d_model, length)
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                Transpose(shape=(1, 2)),        #  (batch, d_model, length) -> (batch, length, d_model)
            )
        elif version == 2:
            self.sequential = nn.Sequential(
                nn.LayerNorm(d_model),          # (batch, length, d_model) -> (batch, length, d_model)
                Transpose(shape=(1, 2)),        # (batch, length, d_model) -> (batch, d_model, length)
                nn.Conv1d(d_model, d_model * 2, 1), # (batch, d_model, length) -> (batch, d_model*2, length)
                nn.GLU(1), # (batch, d_model*2, length) -> (batch, d_model, length)
                nn.Conv1d(d_model, d_model, kernel_size, padding=(kernel_size - 1) // 2, groups=d_model),
                # (batch, d_model, length) -> (batch, d_model, length)
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Conv1d(d_model, d_model, 1),
                nn.Dropout(dropout),
                Transpose(shape=(1, 2)),        #  (batch, d_model, length) -> (batch, length, d_model)
            )

        else:
            raise ValueError("Version should be 1 or 2.")

    def forward(self, inputs):
        return self.sequential(inputs)

def residual_connection_module_creater(module: nn.Module, d_model, *, module_factor, input_factor, **kargs):
    return ResidualConnectionModule(
            module(d_model=d_model, **kargs),
            module_factor=module_factor,
            input_factor=input_factor
        )

class ComformerBlock(nn.Module):
    def __init__(self, *, d_model, **kargs):
        super().__init__()

        self.feed_forward_1 = residual_connection_module_creater(FeedForwardModule, d_model, **kargs["feedforward"])
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                **kargs["transformer"]
            ),
            num_layers=1
        )

        self.conv = residual_connection_module_creater(ConvolutionModule, d_model, **kargs["conv"])

        self.feed_forward_2 = residual_connection_module_creater(FeedForwardModule, d_model, **kargs["feedforward"])

    def forward(self, mels):
        """
        args:
            mels: (batch size, length, 40)
        return:
            out: (batch size, n_spks)
        """
        # out: (batch size, length, d_model)
        out = self.feed_forward_1(mels)
        out = self.encoder(out)
        out = self.conv(out)
        out = self.feed_forward_2(out)
        
        return out

class DimSumReducer(nn.Module):
    def __init__(self, dim):
        super(DimSumReducer, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return torch.sum(x, dim=self.dim)

class DimMeanReducer(nn.Module):
    def __init__(self, dim):
        super(DimSumReducer, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return torch.mean(x, dim=self.dim)

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim, reduce_type="sum"):
        super(SelfAttentionPooling, self).__init__()
        # self.W = nn.Linear(input_dim, 1, bias=False)
        self.W = torch.nn.Parameter(torch.empty(input_dim, 1), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.W)

        if reduce_type == "mean":
            self.reducer = DimMeanReducer(1)
        elif reduce_type =="sum":
            self.reducer = DimSumReducer(1)
        else:
            raise ValueError("Undifined reducer type!")
        
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)

        """
        # att_w = nn.functional.softmax(self.W(batch_rep), dim=1)
        att_w = nn.functional.softmax(batch_rep @ self.W, dim=1)
        utter_rep = self.reducer(batch_rep * att_w)
        return utter_rep

class Comformer(nn.Module):
    def __init__(self, *, input_mels, d_model, post_comformer_dropout=0.1, pred_layer=0, comformer_conf, **kargs):
        super().__init__()

        self.input_mels = input_mels

        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(input_mels, d_model)

        self.layers = nn.ModuleList([ComformerBlock(
            d_model=d_model,
            **comformer_conf["submodules"]
        ) for _ in range(comformer_conf["layers"])])

        self.post_comformer_drop = nn.Dropout(post_comformer_dropout)

        self.self_attention_pooling = SelfAttentionPooling(d_model)

        if comformer_conf["norm_after_cf_block"]:
            self.layer_norm = nn.LayerNorm(d_model)
        else:
            self.layer_norm = nn.Identity()

        if pred_layer > 0:
            self.pred_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU()
                ) for _ in range(pred_layer)]
            )
        else:
            self.pred_layers = nn.ModuleList([nn.Identity()])

    def forward(self, mels):

        out = self.prenet(mels)

        for layer in self.layers:
            out = layer(out)

        out = self.post_comformer_drop(out)
        out = self.layer_norm(out)
        stats = self.self_attention_pooling(out)

        # out: (batch, n_spks)
        for pred_layer in self.pred_layers:
            stats = pred_layer(stats)
            
        return stats
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def summerize(self, len=128):
        summary(self, (1, len, self.input_mels), device=self.device, depth=7)
