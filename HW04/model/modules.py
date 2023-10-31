import torch
import torch.nn as nn
from torchinfo import summary

import logging
logger = logging.getLogger(__name__)

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
    outputs = (module(inputs) * module_factor + inputs * input_factor)
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
    def __init__(self, d_model, version: int = 2, dropout: float = 0.1, **kargs):
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
    def __init__(self, d_model, version: int = 2, kernel_size: int = 31, dropout: float = 0.1, **kargs):
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, **kargs):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.self_attn(x, x, x, need_weights=False)[0]
        return x

class ComformerBlock(nn.Module):
    def __init__(self, d_model, **kargs):
        super().__init__()

        self.feed_forward_1 = ResidualConnectionModule(
            FeedForwardModule(d_model, **kargs["feedforward"]),
            **kargs["feedforward"]["residual_connection"]
        )

        self.self_attn = ResidualConnectionModule(
            MultiheadSelfAttention(d_model, **kargs["multiheadAttention"]),
            **kargs["multiheadAttention"]["residual_connection"]
        )

        self.conv = ResidualConnectionModule(
            ConvolutionModule(d_model, **kargs["conv"]),
            **kargs["conv"]["residual_connection"]
        )

        self.feed_forward_2 = ResidualConnectionModule(
            FeedForwardModule(d_model, **kargs["feedforward"]),
            **kargs["feedforward"]["residual_connection"]
        )

    def forward(self, mels):
        """
        args:
            mels: (batch size, length, 40)
        return:
            out: (batch size, n_spks)
        """
        # out: (batch size, length, d_model)
        out = self.feed_forward_1(mels)
        out = self.self_attn(out)
        out = self.conv(out)
        out = self.feed_forward_2(out)
        out = nn.functional.layer_norm(out, mels.shape[1:])
        return out

class DimSumReducer(nn.Module):
    def __init__(self, dim):
        super(DimSumReducer, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return torch.sum(x, dim=self.dim)

class DimMeanReducer(nn.Module):
    def __init__(self, dim):
        super(DimMeanReducer, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return torch.mean(x, dim=self.dim)

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim, reducer="sum"):
        super(SelfAttentionPooling, self).__init__()
        # self.W = nn.Linear(input_dim, 1, bias=False)
        self.W = torch.nn.Parameter(torch.empty(input_dim, 1), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.W)

        if reducer == "mean":
            self.reducer = DimMeanReducer(1)
        elif reducer =="sum":
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
