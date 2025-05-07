from models.parametric_models.ParametricConvNet import ParametricConvNet

class SmallConvNet(ParametricConvNet):
    def __init__(self):
        """
        Small Convolutional Neural Network with approximately 10,000 parameters.
        
        Architecture:
        - 2 Conv layers with small number of filters
        - Small fully connected layers
        - No dropout to reduce parameters
        
        Parameter count breakdown:
        - Conv1: 1->8 channels, 3x3 kernel: 8*(1*3*3+1) = 80 parameters
        - Conv2: 8->16 channels, 3x3 kernel: 16*(8*3*3+1) = 1,168 parameters
        - First FC layer: (16*7*7)->32: 32*(16*7*7+1) = 8,224 parameters
          (After 2 pooling layers of size 2, the 28x28 input becomes 7x7)

        - Output FC layer: 32->10: 10*(32+1) = 330 parameters
        
        Total: ~9,802 parameters
        """
        super(SmallConvNet, self).__init__(
            conv_channels=[8, 8],       # Small number of filters
            kernel_sizes=[3, 3],         # Standard 3x3 kernels
            padding=[1, 1],              # Preserve spatial dimensions
            pool_sizes=[2, 2],           # Reduce spatial dimensions by 4x
            fc_sizes=[16],               # Small hidden layer
            dropout_rate=0               # No dropout to reduce parameter count
        )
        
    def get_name(self):
        return "SmallConvNet" 