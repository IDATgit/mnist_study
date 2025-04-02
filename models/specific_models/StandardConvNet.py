from models.parametric_models.ParametricConvNet import ParametricConvNet

class StandardConvNet(ParametricConvNet):
    def __init__(self):
        super(StandardConvNet, self).__init__(
            conv_channels=[32, 64, 64],
            kernel_sizes=[3, 3, 3],
            padding=[1, 1, 1],
            pool_sizes=[2, 2, 1],
            fc_sizes=[512, 128],
            dropout_rate=0
        )
        
    def get_name(self):
        return "StandardConvNet" 