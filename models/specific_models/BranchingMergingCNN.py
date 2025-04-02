import torch
import torch.nn as nn
import torch.nn.functional as F

class HomogeneousVectorCapsule(nn.Module):
    """Homogeneous Vector Capsule layer that uses element-wise multiplication.
    Based on the original implementation from https://github.com/AdamByerly/BMCNNwHFCs"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(HomogeneousVectorCapsule, self).__init__()
        # Original implementation uses conv2d with bias=False
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        # Original implementation: conv -> bn -> square
        x = self.bn(self.conv(x))
        x = x * x  # Element-wise squaring as per paper
        return x
    
    def get_name(self):
        return "BranchingMergingCNN"
    
class BranchingMergingCNN(nn.Module):
    """Branching/Merging CNN with Homogeneous Vector Capsules.
    Based on the original implementation from https://github.com/AdamByerly/BMCNNwHFCs"""
    def __init__(self):
        super(BranchingMergingCNN, self).__init__()
        
        # Initial convolution (matches original: 32 filters, 3x3 kernel)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Branch 1 (matches original: two HVC layers with 64 filters each)
        self.branch1_conv1 = HomogeneousVectorCapsule(32, 64, kernel_size=3, padding=1)
        self.branch1_conv2 = HomogeneousVectorCapsule(64, 64, kernel_size=3, padding=1)
        
        # Branch 2 (matches original: two HVC layers with 64 filters each)
        self.branch2_conv1 = HomogeneousVectorCapsule(32, 64, kernel_size=3, padding=1)
        self.branch2_conv2 = HomogeneousVectorCapsule(64, 64, kernel_size=3, padding=1)
        
        # Merge and final layers (matches original: one HVC layer with 128 filters)
        self.merge_conv = HomogeneousVectorCapsule(128, 128, kernel_size=3, padding=1)
        
        # Global average pooling (matches original implementation)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final classification layer (matches original: dense layer to 10 classes)
        self.fc = nn.Linear(128, 10, bias=False)
        
    def forward(self, x):
        # Initial convolution (matches original: conv -> bn -> relu)
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Branch 1 (matches original: two HVC layers with relu)
        b1 = F.relu(self.branch1_conv1(x))
        b1 = F.relu(self.branch1_conv2(b1))
        
        # Branch 2 (matches original: two HVC layers with relu)
        b2 = F.relu(self.branch2_conv1(x))
        b2 = F.relu(self.branch2_conv2(b2))
        
        # Merge branches (matches original: concatenate -> HVC -> relu)
        x = torch.cat([b1, b2], dim=1)
        x = F.relu(self.merge_conv(x))
        
        # Global average pooling (matches original)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Final classification (matches original)
        x = self.fc(x)
        return x 