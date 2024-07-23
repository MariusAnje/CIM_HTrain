import torch
from torch import nn
from modules import NModel
from modules import CrossLinear, CrossConv2d

class HModel(NModel):
    def __init__(self, model_name=None, device_type="RRAM1"):
        super().__init__(model_name, device_type)
        self.A_modules = []
        self.D_modules = []
    
    def zero_init(self):
        for m in self.D_modules:
            m.op.weight.data = torch.zeros_like(m.op.weight)
            if m.op.bias is not None:
                m.op.bias.data = torch.zeros_like(m.op.bias)
    
    @torch.no_grad()
    def id_init(self):
        for m in self.D_modules:
            num_channels = m.op.weight.shape[1]
            if m.op.weight.shape[2] == 1:
                m.op.weight.data = torch.eye(num_channels).view(num_channels, num_channels, 1, 1).to(m.op.weight.device)
            elif m.op.weight.shape[2] == 3:
                with torch.no_grad():
                    m.op.weight.zero_()
                    for i in range(num_channels):
                        m.op.weight.data[i, i, 1, 1] = 1.0
            if m.op.bias is not None:
                m.op.bias.data = torch.zeros_like(m.op.bias)
    
    def continue_setup(self):
        for m in self.A_modules:
            m.op.weight.requires_grad = False
            m.op.bias.requires_grad = False
        self.zero_init()
    
    def set_analog_noise_multiple(self, noise_type, dev_var, rate_max=0, rate_zero=0, write_var=0, **kwargs):
        for mo in self.A_modules:
            mo.set_noise_multiple(noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)

# class CIFAR(HModel):
#     def __init__(self, device_type="RRAM1"):
#         super().__init__("CIFAR", device_type)
#         # self.N_weight=6
#         # self.N_ADC=6
#         # self.array_size=64
#         self.fix_d = True

#         self.conv1 = self.get_conv2d(3, 64, 3, padding=1)
#         self.conv2 = self.get_conv2d(64, 64, 3, padding=1)
#         self.conv2_f = self.get_conv2d(64, 64, 1,bias=False)
#         self.pool1 = nn.MaxPool2d(2,2)

#         self.conv3 = self.get_conv2d(64,128,3, padding=1)
#         self.conv4 = self.get_conv2d(128,128,3, padding=1)
#         self.conv4_f = self.get_conv2d(128,128,1,bias=False)
#         self.pool2 = nn.MaxPool2d(2,2)

#         self.conv5 = self.get_conv2d(128,256,3, padding=1)
#         self.conv6 = self.get_conv2d(256,256,3, padding=1)
#         self.conv6_f = self.get_conv2d(256,256,1,bias=False)
#         self.pool3 = nn.MaxPool2d(2,2)
        
#         self.fc1 = self.get_linear(256 * 4 * 4, 1024)
#         self.fc2 = self.get_linear(1024, 1024)
#         self.fc3 = self.get_linear(1024, 10)
#         self.relu = nn.ReLU()
#         self.D_modules = [self.conv2_f, self.conv4_f, self.conv6_f]
#         self.A_modules = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]
#         self.zero_init()

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         if not self.fix_d:
#             x_f = self.conv2_f(x)
#             # x_f = self.relu(x_f)
#         x = self.conv2(x)
#         if not self.fix_d:
#             x = x + x_f
#         x = self.relu(x)
#         x = self.pool1(x)

#         x = self.conv3(x)
#         x = self.relu(x)
#         if not self.fix_d:
#             x_f = self.conv4_f(x)
#             # x_f = self.relu(x_f)
#         x = self.conv4(x)
#         if not self.fix_d:
#             x = x + x_f
#         x = self.relu(x)
#         x = self.pool2(x)

#         x = self.conv5(x)
#         x = self.relu(x)
#         if not self.fix_d:
#             x_f = self.conv6_f(x)
#             # x_f = self.relu(x_f)
#         x = self.conv6(x)
#         if not self.fix_d:
#             x = x + x_f
#         x = self.relu(x)
#         x = self.pool3(x)

#         x = self.unpack_flattern(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         return x


class CIFAR(HModel):
    def continue_setup(self):
        for m in self.A_modules:
            m.op.weight.requires_grad = False
            m.op.bias.requires_grad = False
        self.id_init()

    def __init__(self, device_type="RRAM1"):
        super().__init__("CIFAR", device_type)
        # self.N_weight=6
        # self.N_ADC=6
        # self.array_size=64
        self.fix_d = True

        self.conv1 = self.get_conv2d(3, 64, 3, padding=1)
        self.conv1_f = self.get_conv2d(64, 64, 3, padding=1,bias=False)
        self.conv2 = self.get_conv2d(64, 64, 3, padding=1)
        self.conv2_f = self.get_conv2d(64, 64, 3, padding=1,bias=False)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv3 = self.get_conv2d(64,128,3, padding=1)
        self.conv3_f = self.get_conv2d(128,128,3, padding=1,bias=False)
        self.conv4 = self.get_conv2d(128,128,3, padding=1)
        self.conv4_f = self.get_conv2d(128,128,3, padding=1,bias=False)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv5 = self.get_conv2d(128,256,3, padding=1)
        self.conv5_f = self.get_conv2d(256,256,3, padding=1,bias=False)
        self.conv6 = self.get_conv2d(256,256,3, padding=1)
        self.conv6_f = self.get_conv2d(256,256,3, padding=1,bias=False)
        self.pool3 = nn.MaxPool2d(2,2)
        
        self.fc1 = self.get_linear(256 * 4 * 4, 1024)
        self.fc2 = self.get_linear(1024, 1024)
        self.fc3 = self.get_linear(1024, 10)
        self.relu = nn.ReLU()
        self.D_modules = [self.conv1_f, self.conv2_f, self.conv3_f, self.conv4_f, self.conv5_f, self.conv6_f]
        self.A_modules = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]
        self.id_init()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        if self.fix_d:
            x_f = x
        else:
            x_f = self.conv2_f(x)
        x = self.conv2(x)
        x = x + x_f
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu(x)
        if self.fix_d:
            x_f = x
        else:
            x_f = self.conv4_f(x)
        x = self.conv4(x)
        x = x + x_f
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.relu(x)
        if self.fix_d:
            x_f = x
        else:
            x_f = self.conv6_f(x)
        x = self.conv6(x)
        x = x + x_f
        x = self.relu(x)
        x = self.pool3(x)

        x = self.unpack_flattern(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x