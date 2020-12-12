# -*- coding: UTF-8 -*-
"""
-------------------------------------------------
# @Project -> File   ：JointCwsPosParser -> layer
# @Author ：bosskai
# @Date   ：2020/7/21 16:11
# @Email  ：19120406@bjtu.edu.cn
-------------------------------------------------
"""
import torch

from torch import nn
from fastNLP.modules.dropout import TimestepDropout


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0):
        super(MLP, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = TimestepDropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class Biaffine(nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s

class Triaffine(nn.Module):

    def __init__(self, n_in, bias_x=False, bias_y=False):
        super(Triaffine, self).__init__()

        self.n_in = n_in
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_in + bias_x,
                                                n_in,
                                                n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y, z):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        w = torch.einsum('bzk,ikj->bzij', z, self.weight)
        # [batch_size, seq_len, seq_len, seq_len]
        s = torch.einsum('bxi,bzij,byj->bzxy', x, w, y)

        return s
