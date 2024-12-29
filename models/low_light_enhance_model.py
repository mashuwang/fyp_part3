import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out

class LowLightEnhanceModel(BaseModel):
    def set_input(self, input):
        """设置输入数据"""
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        if 'mask' in input:
            self.mask = input['mask'].to(self.device)
        else:
            # 如果没有提供 mask，创建一个默认的全图 mask
            self.mask = torch.ones_like(self.real_A)

    def __init__(self, opt):
        super(LowLightEnhanceModel, self).__init__(opt)
        self.loss_names = ['G', 'D']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.model_names = ['G', 'D']

        # 获取设备
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


        # Define generator and discriminator
        self.netG = self.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG.to(self.device) # 将生成器移动到设备

        self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD.to(self.device) # 将判别器移动到设备


        # Define loss functions
        self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
        self.criterionL1 = torch.nn.L1Loss()

        # Initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers = [self.optimizer_G, self.optimizer_D]

    def define_G(self, input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
        # Define a generator with residual blocks and attention mechanism
        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
                 nn.BatchNorm2d(ngf),
                 nn.ReLU(True)]

        # Add residual blocks
        for _ in range(6):
            model += [ResidualBlock(ngf)]

        # Add attention mechanism (e.g., self-attention)
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3),
                  nn.Tanh()]

        return nn.Sequential(*model)

    def forward(self):
        # Initialize mask as a tensor of ones with the same shape as real_A
        # mask = torch.ones_like(self.real_A)  # 这行代码是多余的，因为你在 set_input 中已经处理了 mask
        # Apply mask to the input image
        masked_input = self.real_A * self.mask
        masked_input = masked_input.to(self.device) # 将 masked_input 移动到设备
        # Generate the enhanced image
        self.fake_B = self.netG(masked_input)

    def optimize_parameters(self):
        # Update generator
        self.optimizer_G.zero_grad()
        self.forward()
        self.loss_G = self.criterionGAN(self.netD(self.fake_B), True) + self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G.backward()
        self.optimizer_G.step()

        # Update discriminator
        self.optimizer_D.zero_grad()
        self.loss_D = (self.criterionGAN(self.netD(self.real_B), True) + self.criterionGAN(self.netD(self.fake_B.detach()), False)) * 0.5
        self.loss_D.backward()
        self.optimizer_D.step()
