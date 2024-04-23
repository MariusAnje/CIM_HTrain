import torch
from torch import autograd
from torch import nn
from Functions import QuantFunction, sepMM, sepConv2d
from noise import set_noise_multiple
import numpy as np

class Quant(nn.Module):
    def __init__(self, N, running=True):
        super().__init__()
        self.running = running
        self.register_buffer('running_range', torch.ones(1))
        self.running_range
        self.N = N
        self.func = QuantFunction.apply
        self.momentum = 0.9
    
    def forward(self, x):
        if self.running:
            if self.training:
                if self.running_range == 0:
                    self.running_range += x.abs().max().item()
                else:
                    self.running_range = self.running_range * self.momentum + x.abs().max().item() * (1-self.momentum)
            return self.func(self.N, x, self.running_range)
        else:
            return self.func(self.N, x)


class NModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = 1
        self.fast = False

    def set_noise_multiple(self, noise_type, dev_var, rate_max=0, rate_zero=0, write_var=0, **kwargs):
        set_noise_multiple(self, noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
    
    def clear_noise(self):
        self.noise = torch.zeros_like(self.op.weight)
    
    def clear_mask(self):
        self.noise = torch.ones_like(self.op.weight)

    def normalize(self):
        if self.original_w is None:
            self.original_w = self.op.weight.data
        if (self.original_b is None) and (self.op.bias is not None):
            self.original_b = self.op.bias.data
        scale = self.op.weight.data.abs().max().item()
        self.scale = scale
        self.op.weight.data = self.op.weight.data / scale

    def denormalize(self):
        if self.original_w is not None:
            self.scale = 1
            self.op.weight.data = self.original_w.data
            self.original_w = None
        if self.original_b is not None:
            self.op.bias.data = self.original_b.data
            self.original_b = None

class CrossLinear(NModule):
    def __init__(self, in_features, out_features, bias=True, N_weight=4, N_ADC=4, array_size=32) -> None:
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.register_buffer('noise', torch.zeros_like(self.op.weight))
        self.register_buffer('mask', torch.ones_like(self.op.weight))
        self.running_act = None
        self.q_w_f = Quant(N_weight, False)
        self.q_a_train = Quant(N_ADC, True)
        array_number = int(np.ceil(in_features / array_size)) # not exactly this meaning but close
        self.q_a_f = nn.ModuleList([Quant(N_ADC, True) for _ in range(array_number)])
        self.array_size = array_size
    
    def forward(self, x):
        if self.fast:
            x = self.q_a_train(nn.functional.linear(x, self.q_w_f(self.op.weight) + self.noise))
            # for m in self.q_a_f:
            #     m.running_range.data = self.q_a_train.running_range.data
        else:
            x = sepMM(x, self.q_w_f(self.op.weight) + self.noise, self.q_a_f, self.array_size)
        if self.op.bias is not None:
            return x + self.op.bias
        else:
            return x

class CrossConv2d(NModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', N_weight=4, N_ADC=4, array_size=32):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.register_buffer('noise', torch.zeros_like(self.op.weight))
        self.register_buffer('mask', torch.zeros_like(self.op.weight))
        self.function = nn.functional.conv2d
        self.scale = 1.0
        self.running_act = None
        self.q_w_f = Quant(N_weight, False)
        self.q_a_train = Quant(N_ADC, True)
        array_number = int(np.ceil(in_channels / array_size)) # not exactly this meaning but close
        self.q_a_f = nn.ModuleList([nn.ModuleList([nn.ModuleList([Quant(N_ADC, True) for _ in range(self.op.kernel_size[1])]) for _ in range(self.op.kernel_size[0])]) for _ in range(array_number)])
        self.array_size = array_size

    def forward(self, x):
        if self.fast:
            x = self.q_a_train(nn.functional.conv2d(x, self.q_w_f(self.op.weight) + self.noise, padding=self.op.padding, stride=self.op.stride))
            # for m in self.q_a_f:
            #     for k in m:
            #         for v in k:
            #             v.running_range.data = self.q_a_train.running_range.data
        else:
            x = sepConv2d(x, self.q_w_f(self.op.weight) + self.noise, self.q_a_f, self.array_size, padding=self.op.padding)
        if self.op.bias is not None:
            x += self.op.bias.reshape(1,-1,1,1).expand_as(x)
        return x
        
def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

class NModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.original_w = None
        self.original_b = None
    
    def set_noise_multiple(self, noise_type, dev_var, rate_max=0, rate_zero=0, write_var=0, **kwargs):
        for mo in self.modules():
            if isinstance(mo, NModule):
                mo.set_noise_multiple(noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)

    def clear_noise(self):
        for m in self.modules():
            if isinstance(m, NModule):
                m.clear_noise()
    
    def make_fast(self):
        for m in self.modules():
            if isinstance(m, NModule):
                m.fast = True
    
    def make_slow(self):
        for m in self.modules():
            if isinstance(m, NModule):
                m.fast = False
    
    def clear_mask(self):
        for m in self.modules():
            if isinstance(m, NModule):
                m.clear_mask()
    
    def de_normalize(self):
        for mo in self.modules():
            if isinstance(mo, NModule):
                mo.denormalize()
    
    def normalize(self):
        for mo in self.modules():
            if isinstance(mo, NModule):
                mo.normalize()
    
    def unpack_flattern(self, x):
        return x.view(-1, num_flat_features(x))
    
    def get_conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        return CrossConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, N_weight=self.N_weight,N_ADC=self.N_ADC,array_size=self.array_size)

    def get_linear(self, in_features, out_features, bias=True):
        return CrossLinear(in_features, out_features, bias, N_weight=self.N_weight,N_ADC=self.N_ADC,array_size=self.array_size)
