import torch
from torch import nn
from modules import NModel
from modules import CrossLinear, CrossConv2d, LoCrossConv2d, LoCrossLinear

class PModel(NModel):
    def __init__(self, model_name=None, device_type="RRAM1"):
        super().__init__(model_name, device_type)
    
    def get_conv2d(self, rank, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        return LoCrossConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, 
                           N_weight=self.N_weight,N_ADC=self.N_ADC,array_size=self.array_size,mapping=self.mapping,
                           rank=rank)

    def get_linear(self, rank, in_features, out_features, bias=True):
        return LoCrossLinear(in_features, out_features, bias, 
                           N_weight=self.N_weight,N_ADC=self.N_ADC,array_size=self.array_size,mapping=self.mapping,
                           rank=rank)
    
    def is_Lo_layer(self, m):
        return isinstance(m, LoCrossLinear) or isinstance(m, LoCrossConv2d)

    def zero_init_Lo(self):
        for m in self.modules():
            if self.is_Lo_layer(m):
                m.A.zero_()
                if isinstance(m.B, nn.Parameter):
                    m.B.zero_()
    
    def set_Lo_grad(self, option:bool):
        for m in self.modules():
            if self.is_Lo_layer(m):
                m.A.requires_grad = option
                if isinstance(m.B, nn.Parameter):
                    m.B.requires_grad = option

    def set_normal_grad(self, option:bool):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.requires_grad = option

    def pre_training(self):
        self.set_Lo_grad(False)
        self.zero_init_Lo()
        self.set_normal_grad(True)

    def Lo_only(self):
        self.zero_init_Lo()
        self.set_Lo_grad(True)
        self.set_normal_grad(False)
    
    def get_Lo_parameters(self):
        parameter_list = []
        for m in self.modules():
            if self.is_Lo_layer(m):
                parameter_list.append(m.A)
                if isinstance(m.B, nn.Parameter):
                    parameter_list.append(m.B)
        return parameter_list

class CIFAR(PModel):
    def __init__(self, device_type="RRAM1"):
        super().__init__("CIFAR", device_type)
        # self.N_weight=6
        # self.N_ADC=6
        # self.array_size=64
        rank = 0

        self.conv1 = self.get_conv2d(0, 3, 64, 3, padding=1)
        self.conv2 = self.get_conv2d(rank, 64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv3 = self.get_conv2d(rank, 64,128,3, padding=1)
        self.conv4 = self.get_conv2d(rank, 128,128,3, padding=1)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv5 = self.get_conv2d(rank, 128,256,3, padding=1)
        self.conv6 = self.get_conv2d(rank, 256,256,3, padding=1)
        self.pool3 = nn.MaxPool2d(2,2)
        
        self.fc1 = self.get_linear(rank, 256 * 4 * 4, 1024)
        self.fc2 = self.get_linear(rank, 1024, 1024)
        self.fc3 = self.get_linear(0, 1024, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.unpack_flattern(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x