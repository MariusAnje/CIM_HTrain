import torch
from torch import nn
from modules import NModel
from modules import CrossLinear, CrossConv2d

def res_f(x, module, res_module, fix_d):
    if fix_d:
        x_f = 0
    else:
        x_f = res_module(x)
    x = module(x)
    return x + x_f

class HModel(NModel):
    def __init__(self, model_name=None, device_type="RRAM1"):
        super().__init__(model_name, device_type)
        self.A_modules = []
        self.D_modules = []
    
    @torch.no_grad()
    def zero_init(self):
        for m in self.D_modules:
            m.op.weight.data = torch.zeros_like(m.op.weight)
            if m.op.bias is not None:
                m.op.bias.data = torch.zeros_like(m.op.bias)
    
    @torch.no_grad()
    def id_init(self):
        for m in self.D_modules:
            if len(m.op.weight.shape) == 4:
                num_channels = m.op.weight.shape[1]
                if m.op.weight.shape[2] == 1:
                    m.op.weight.data = torch.eye(num_channels).view(num_channels, num_channels, 1, 1).to(m.op.weight.device)
                elif m.op.weight.shape[2] == 3:
                    with torch.no_grad():
                        m.op.weight.zero_()
                        for i in range(num_channels):
                            m.op.weight.data[i, i, 1, 1] = 1.0
            else:
                m.op.weight.data = torch.zeros_like(m.op.weight)
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

class CIFAR_Plain(HModel):
    def __init__(self, device_type="RRAM1"):
        super().__init__("CIFAR", device_type)
        # self.N_weight=6
        # self.N_ADC=6
        # self.array_size=64
        self.fix_d = True
        fixing_size = 1
        p_size = fixing_size // 2

        self.conv1 = self.get_conv2d(3, 64, 3, padding=1)
        self.conv1_f = self.get_conv2d(3, 64, fixing_size, padding=p_size)
        self.conv2 = self.get_conv2d(64, 64, 3, padding=1)
        # self.conv2_f = self.get_conv2d(64, 64, 3,bias=False)
        self.conv2_f = self.get_conv2d(64, 64, fixing_size, padding=p_size)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv3 = self.get_conv2d(64,128,3, padding=1)
        self.conv3_f = self.get_conv2d(64,128, fixing_size, padding=p_size)
        self.conv4 = self.get_conv2d(128,128,3, padding=1)
        # self.conv4_f = self.get_conv2d(128,128,1,bias=False)
        self.conv4_f = self.get_conv2d(128,128, fixing_size, padding=p_size)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv5 = self.get_conv2d(128,256,3, padding=1)
        self.conv5_f = self.get_conv2d(128,256, fixing_size, padding=p_size)
        self.conv6 = self.get_conv2d(256,256,3, padding=1)
        # self.conv6_f = self.get_conv2d(256,256,1,bias=False)
        self.conv6_f = self.get_conv2d(256,256, fixing_size, padding=p_size)
        self.pool3 = nn.MaxPool2d(2,2)
        
        self.fc1 = self.get_linear(256 * 4 * 4, 1024)
        self.fc1_f = self.get_linear(256 * 4 * 4, 1024)
        self.fc2 = self.get_linear(1024, 1024)
        self.fc2_f = self.get_linear(1024, 1024)
        self.fc3 = self.get_linear(1024, 10)
        self.fc3_f = self.get_linear(1024, 10)
        self.relu = nn.ReLU()
        self.D_modules = [self.conv1_f, self.conv2_f, self.conv3_f, self.conv4_f, self.conv5_f, self.conv6_f, self.fc1_f, self.fc2_f, self.fc3_f]
        self.A_modules = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.fc1, self.fc2, self.fc3]
        self.zero_init()

    def forward(self, x):
        def select_fix(x, conv, i, module, module_f):
            if i in conv:
                x = res_f(x, module, module_f, self.fix_d)
            else:
                x = module(x)
            return x

        conv = [2,4,6]
        fc = [1,2,3]
        x = select_fix(x, conv, 1, self.conv1, self.conv1_f)
        x = self.relu(x)
        x = select_fix(x, conv, 2, self.conv2, self.conv2_f)
        x = self.relu(x)
        x = self.pool1(x)

        x = select_fix(x, conv, 3, self.conv3, self.conv3_f)
        x = self.relu(x)
        x = select_fix(x, conv, 4, self.conv4, self.conv4_f)
        x = self.relu(x)
        x = self.pool2(x)

        x = select_fix(x, conv, 5, self.conv5, self.conv5_f)
        x = self.relu(x)
        x = select_fix(x, conv, 6, self.conv6, self.conv6_f)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.unpack_flattern(x)
        x = select_fix(x, fc, 1, self.fc1, self.fc1_f)
        x = self.relu(x)
        x = select_fix(x, fc, 2, self.fc2, self.fc2_f)
        x = self.relu(x)
        x = select_fix(x, fc, 3, self.fc3, self.fc3_f)
        return x

class CIFAR_Res(HModel):
    def __init__(self, device_type="RRAM1"):
        super().__init__("CIFAR", device_type)
        # self.N_weight=6
        # self.N_ADC=6
        # self.array_size=64
        self.fix_d = True
        fixing_size = 1
        p_size = fixing_size // 2

        self.conv1 = self.get_conv2d(3, 64, 3, padding=1)
        self.conv1_f = self.get_conv2d(3, 64, fixing_size, padding=p_size)
        self.conv2 = self.get_conv2d(64, 64, 3, padding=1)
        # self.conv2_f = self.get_conv2d(64, 64, 3,bias=False)
        self.conv2_f = self.get_conv2d(64, 64, fixing_size, padding=p_size)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv3 = self.get_conv2d(64,128,3, padding=1)
        self.conv3_f = self.get_conv2d(64,128, fixing_size, padding=p_size)
        self.conv4 = self.get_conv2d(128,128,3, padding=1)
        # self.conv4_f = self.get_conv2d(128,128,1,bias=False)
        self.conv4_f = self.get_conv2d(128,128, fixing_size, padding=p_size)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv5 = self.get_conv2d(128,256,3, padding=1)
        self.conv5_f = self.get_conv2d(128,256, fixing_size, padding=p_size)
        self.conv6 = self.get_conv2d(256,256,3, padding=1)
        # self.conv6_f = self.get_conv2d(256,256,1,bias=False)
        self.conv6_f = self.get_conv2d(256,256, fixing_size, padding=p_size)
        self.pool3 = nn.MaxPool2d(2,2)
        
        self.fc1 = self.get_linear(256 * 4 * 4, 1024)
        self.fc1_f = self.get_linear(256 * 4 * 4, 1024)
        self.fc2 = self.get_linear(1024, 1024)
        self.fc2_f = self.get_linear(1024, 1024)
        self.fc3 = self.get_linear(1024, 10)
        self.fc3_f = self.get_linear(1024, 10)
        self.relu = nn.ReLU()
        self.D_modules = [self.conv1_f, self.conv2_f, self.conv3_f, self.conv4_f, self.conv5_f, self.conv6_f, self.fc1_f, self.fc2_f, self.fc3_f]
        self.A_modules = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.fc1, self.fc2, self.fc3]
        self.id_init()
    
    def continue_setup(self):
        for m in self.A_modules:
            m.op.weight.requires_grad = False
            m.op.bias.requires_grad = False
        self.id_init()
    
    def res_f(self, x, module, res_module, fix_d):
        if fix_d:
            x_f = x
        else:
            x_f = res_module(x)
        x = module(x)
        return x + x_f

    def forward(self, x):
        def select_fix(x, conv, i, module, module_f):
            if i in conv:
                x = self.res_f(x, module, module_f, self.fix_d)
            else:
                x = module(x)
            return x

        conv = [2,4,6]
        fc = [1,2,3]
        x = select_fix(x, conv, 1, self.conv1, self.conv1_f)
        x = self.relu(x)
        x = select_fix(x, conv, 2, self.conv2, self.conv2_f)
        x = self.relu(x)
        x = self.pool1(x)

        x = select_fix(x, conv, 3, self.conv3, self.conv3_f)
        x = self.relu(x)
        x = select_fix(x, conv, 4, self.conv4, self.conv4_f)
        x = self.relu(x)
        x = self.pool2(x)

        x = select_fix(x, conv, 5, self.conv5, self.conv5_f)
        x = self.relu(x)
        x = select_fix(x, conv, 6, self.conv6, self.conv6_f)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.unpack_flattern(x)
        x = select_fix(x, fc, 1, self.fc1, self.fc1_f)
        x = self.relu(x)
        x = select_fix(x, fc, 2, self.fc2, self.fc2_f)
        x = self.relu(x)
        x = select_fix(x, fc, 3, self.fc3, self.fc3_f)
        return x


class CIFAR_Seq(HModel):
    def __init__(self, device_type="RRAM1"):
        super().__init__("CIFAR", device_type)
        # self.N_weight=6
        # self.N_ADC=6
        # self.array_size=64
        self.fix_d = True
        fixing_size = 1
        p_size = fixing_size // 2

        self.conv1 = self.get_conv2d(3, 64, 3, padding=1)
        self.conv1_f = self.get_conv2d(64, 64, fixing_size, padding=p_size)
        self.conv2 = self.get_conv2d(64, 64, 3, padding=1)
        # self.conv2_f = self.get_conv2d(64, 64, 3,bias=False)
        self.conv2_f = self.get_conv2d(64, 64, fixing_size, padding=p_size)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv3 = self.get_conv2d(64,128,3, padding=1)
        self.conv3_f = self.get_conv2d(128,128, fixing_size, padding=p_size)
        self.conv4 = self.get_conv2d(128,128,3, padding=1)
        # self.conv4_f = self.get_conv2d(128,128,1,bias=False)
        self.conv4_f = self.get_conv2d(128,128, fixing_size, padding=p_size)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv5 = self.get_conv2d(128,256,3, padding=1)
        self.conv5_f = self.get_conv2d(256,256, fixing_size, padding=p_size)
        self.conv6 = self.get_conv2d(256,256,3, padding=1)
        # self.conv6_f = self.get_conv2d(256,256,1,bias=False)
        self.conv6_f = self.get_conv2d(256,256, fixing_size, padding=p_size)
        self.pool3 = nn.MaxPool2d(2,2)
        
        self.fc1 = self.get_linear(256 * 4 * 4, 1024)
        self.fc1_f = self.get_linear(256 * 4 * 4, 1024)
        self.fc2 = self.get_linear(1024, 1024)
        self.fc2_f = self.get_linear(1024, 1024)
        self.fc3 = self.get_linear(1024, 10)
        self.fc3_f = self.get_linear(1024, 10)
        self.relu = nn.ReLU()
        self.D_modules = [self.conv1_f, self.conv2_f, self.conv3_f, self.conv4_f, self.conv5_f, self.conv6_f, self.fc1_f, self.fc2_f, self.fc3_f]
        self.A_modules = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.fc1, self.fc2, self.fc3]
        self.id_init()

    def conv_f(self, x, module, res_module, fix_d):
        if fix_d:
            x = x
        else:
            x = res_module(x)
        x = module(x)
        return x

    def fc_f(self, x, module, res_module, fix_d):
        if fix_d:
            x_f = 0
        else:
            x_f = res_module(x)
        x = module(x)
        return x + x_f

    def continue_setup(self):
        for m in self.A_modules:
            m.op.weight.requires_grad = False
            m.op.bias.requires_grad = False
        self.id_init()

    def forward(self, x):
        def select_fix(x, conv, i, module, module_f):
            if i in conv:
                x = self.conv_f(x, module, module_f, self.fix_d)
            else:
                x = module(x)
            return x
        
        def select_fix_fc(x, conv, i, module, module_f):
            if i in conv:
                x = self.fc_f(x, module, module_f, self.fix_d)
            else:
                x = module(x)
            return x

        conv = [2,4,6]
        fc = [1,2,3]
        x = select_fix(x, conv, 1, self.conv1, self.conv1_f)
        x = self.relu(x)
        x = select_fix(x, conv, 2, self.conv2, self.conv2_f)
        x = self.relu(x)
        x = self.pool1(x)

        x = select_fix(x, conv, 3, self.conv3, self.conv3_f)
        x = self.relu(x)
        x = select_fix(x, conv, 4, self.conv4, self.conv4_f)
        x = self.relu(x)
        x = self.pool2(x)

        x = select_fix(x, conv, 5, self.conv5, self.conv5_f)
        x = self.relu(x)
        x = select_fix(x, conv, 6, self.conv6, self.conv6_f)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.unpack_flattern(x)
        x = select_fix_fc(x, fc, 1, self.fc1, self.fc1_f)
        x = self.relu(x)
        x = select_fix_fc(x, fc, 2, self.fc2, self.fc2_f)
        x = self.relu(x)
        x = select_fix_fc(x, fc, 3, self.fc3, self.fc3_f)
        return x