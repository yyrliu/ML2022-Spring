import torch
import torch.nn as nn
import torch.nn.functional as F

class AMSoftmax(nn.Module):
    # https://github.com/CoinCheung/pytorch-loss
    def __init__(
            self,
            in_feats,
            n_class,
            norm_affine=False,
            feat_norm=True,
            m=0.35,
            s=30
        ):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s

        if feat_norm:
            self.feat_norm_layer = nn.BatchNorm1d(in_feats, affine=norm_affine)
        else:
            self.feat_norm_layer = nn.Identity()

        self.W = torch.nn.Parameter(torch.empty(in_feats, n_class), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        # print(x.size())
        # print(lb.size())
        # assert x.size()[0] == lb.size()[0]

        x = self.feat_norm_layer(x)
        w_normed = F.normalize(self.W, p=2, dim=0)
        costh = torch.mm(x, w_normed)
        
        delt_costh = torch.zeros_like(costh).scatter_(1, lb.unsqueeze(1), self.m)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return costh, loss
