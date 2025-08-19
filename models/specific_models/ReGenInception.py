import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvModule(nn.Module):
    """Basic Conv Module: Convolution -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionModule(nn.Module):
    """Inception Module with parallel branches"""
    def __init__(self, in_channels, ch1x1, ch3x3):
        super(InceptionModule, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = ConvModule(in_channels, ch1x1, kernel_size=1)
        
        # 3x3 convolution branch (direct, no reduction)
        self.branch3x3 = ConvModule(in_channels, ch3x3, kernel_size=3, padding=1)
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        
        # Concatenate along channel dimension
        outputs = torch.cat([branch1x1, branch3x3], dim=1)
        return outputs

class DownsampleModule(nn.Module):
    """Downsample Module: Conv + MaxPool in parallel, then concat"""
    def __init__(self, in_channels, conv_channels):
        super(DownsampleModule, self).__init__()
        
        # Conv branch (3x3, stride 2)
        self.conv_branch = ConvModule(in_channels, conv_channels, kernel_size=3, stride=2, padding=1)
        
        # MaxPool branch (3x3, stride 2)
        self.pool_branch = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        conv_out = self.conv_branch(x)
        pool_out = self.pool_branch(x)
        
        # Concatenate along channel dimension
        outputs = torch.cat([conv_out, pool_out], dim=1)
        return outputs

class ReGenInception(nn.Module):
    """Small Inception Network for CIFAR-10"""
    def __init__(self, num_classes=10):
        super(ReGenInception, self).__init__()
        
        # Initial conv layer for 28x28x3 inputs
        self.initial_conv = ConvModule(3, 96, kernel_size=3, stride=1, padding=1)
        
        # First Inception Module (32+32=64 filters)
        self.inception1 = InceptionModule(96, ch1x1=32, ch3x3=32)
        
        # Second Inception Module (32+48=80 filters) 
        self.inception2 = InceptionModule(64, ch1x1=32, ch3x3=48)
        
        # First Downsample Module (80 filters)
        self.downsample1 = DownsampleModule(80, conv_channels=80)
        
        # Third Inception Module (96+64=160 filters)
        self.inception3 = InceptionModule(160, ch1x1=96, ch3x3=64)
        
        # Fourth Inception Module (80+80=160 filters)
        self.inception4 = InceptionModule(160, ch1x1=80, ch3x3=80)
        
        # Fifth Inception Module (48+96=144 filters)
        self.inception5 = InceptionModule(160, ch1x1=48, ch3x3=96)
        
        # Second Downsample Module (96 filters)
        self.downsample2 = DownsampleModule(144, conv_channels=96)
        
        # sixth Inception Module (176+160=336 filters)
        self.inception6 = InceptionModule(96 + 144, ch1x1=176, ch3x3=160)
        
        # Seventh Inception Module (176+160=336 filters)
        self.inception7 = InceptionModule(176 + 160, ch1x1=176, ch3x3=160)

        # Global Average Pooling (7x7 kernel for global pooling)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final fully connected layer (10-way outputs)
        self.fc = nn.Linear(336, num_classes)
        
    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)  # 28x28x96
        
        # First set of inception modules
        x = self.inception1(x)    # 28x28x64
        x = self.inception2(x)    # 28x28x80
        
        # First downsampling
        x = self.downsample1(x)   # 14x14x160
        
        # Second set of inception modules
        x = self.inception3(x)    # 14x14x160
        x = self.inception4(x)    # 14x14x160
        x = self.inception5(x)    # 14x14x144
        
        # Second downsampling
        x = self.downsample2(x)   # 7x7x240
        
        # Final inception modules
        x = self.inception6(x)    # 7x7x160
        x = self.inception7(x)    # 7x7x336
        
        # Global average pooling
        x = self.global_avg_pool(x)  # 1x1x336
        x = x.view(x.size(0), -1)    # Flatten: 336
        
        # Final classification
        x = self.fc(x)            # 10
        
        return x
    
    def get_name(self):
        return "ReGenInception"
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())
