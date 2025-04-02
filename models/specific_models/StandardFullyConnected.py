from models.parametric_models.ParametricFullyConnected import ParametricFullyConnected

class StandardFullyConnected(ParametricFullyConnected):
    def __init__(self):
        super(StandardFullyConnected, self).__init__(
            layer_sizes=[784, 128, 64, 10]
        )
        
    def get_name(self):
        return "StandardFC" 