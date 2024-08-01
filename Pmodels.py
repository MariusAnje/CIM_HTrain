import torch
from torch import nn
from modules import NModel
from modules import CrossLinear, CrossConv2d, LoCrossConv2d, LoCrossLinear

class PModel(NModel):
    def __init__(self, model_name=None, device_type="RRAM1"):
        super().__init__(model_name, device_type)
    
    def get_conv2d(self, rank, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        print(rank)
        if rank == -1:
            return CrossConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, 
                            N_weight=self.N_weight,N_ADC=self.N_ADC,array_size=self.array_size,mapping=self.mapping)
        else:
            return LoCrossConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, 
                            N_weight=self.N_weight,N_ADC=self.N_ADC,array_size=self.array_size,mapping=self.mapping,
                            rank=rank)

    def get_linear(self, rank, in_features, out_features, bias=True):
        print(rank)
        if rank == -1:
            return CrossLinear(in_features, out_features, bias, 
                            N_weight=self.N_weight,N_ADC=self.N_ADC,array_size=self.array_size,mapping=self.mapping)
        else:
            return LoCrossLinear(in_features, out_features, bias, 
                            N_weight=self.N_weight,N_ADC=self.N_ADC,array_size=self.array_size,mapping=self.mapping,
                            rank=rank)
    
    def is_Lo_layer(self, m):
        return isinstance(m, LoCrossLinear) or isinstance(m, LoCrossConv2d)

    def zero_init_Lo(self):
        for m in self.modules():
            if self.is_Lo_layer(m):
                if isinstance(m.A, nn.Parameter):
                    m.A.data = torch.randn_like(m.A) * 1e-5
                if isinstance(m.B, nn.Parameter):
                    m.B.data = torch.randn_like(m.B) * 1e-5
    
    def set_Lo_grad(self, option:bool):
        for m in self.modules():
            if self.is_Lo_layer(m):
                if isinstance(m.A, nn.Parameter):
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
                if isinstance(m.A, nn.Parameter):
                    parameter_list.append(m.A)
                if isinstance(m.B, nn.Parameter):
                    parameter_list.append(m.B)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    parameter_list.append(m.bias)
        return parameter_list

class CIFAR(PModel):
    def __init__(self, device_type="RRAM1"):
        super().__init__("CIFAR", device_type)
        # self.N_weight=6
        # self.N_ADC=6
        # self.array_size=64
        rank = 32
        # [ 0, 32,  0,  0,  0,  0] 0.8631
        # [ 0, 16,  0,  0,  0,  0] 0.8574
        # [ 0, 16, 32,  0,  0,  0] 0.8560
        # [ 0, 16, 32, 32,  0,  0] 0.8510
        # [ 0, 16, 32, 32, 64,  0] 0.8475
        # [ 0, 16, 32, 32, 64, 64] 0.8403
        # [ 0, 16, 32, 32, 64, 64, 256,   0,   0] 0.8523 / 0.8558
        # [ 0, 16, 32, 32, 64, 64, 256, 256,   0] 0.8498
        # [ 0, 16, 16, 32, 64, 64, 256,   0,   0] 0.8547
        # [ 0, 16, 16, 16, 64, 64, 256,   0,   0] 0.8481
        # [ 0, 08, 16, 16, 64, 64, 256,   0,   0] 0.8511
        # [ 0, 08, 16, 16, 32, 64, 256,   0,   0] 0.8477
        # [ 0, 08, 16, 16, 32, 32, 256,   0,   0] 0.8438
        # [ 0, 08, 16, 16, 48, 48, 256,   0,   0] 0.8486

        self.conv1 = self.get_conv2d(0, 3, 64, 3, padding=1)
        self.conv2 = self.get_conv2d(8, 64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv3 = self.get_conv2d(16, 64,128,3, padding=1)
        self.conv4 = self.get_conv2d(16, 128,128,3, padding=1)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv5 = self.get_conv2d(48, 128,256,3, padding=1)
        self.conv6 = self.get_conv2d(48, 256,256,3, padding=1)
        self.pool3 = nn.MaxPool2d(2,2)
        
        self.fc1 = self.get_linear(256, 256 * 4 * 4, 1024)
        self.fc2 = self.get_linear(0, 1024, 1024)
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