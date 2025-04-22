from models.parametric_models.ParametricFullyConnected import ParametricFullyConnected

class SmallFullyConnected(ParametricFullyConnected):
    def __init__(self):
        super(SmallFullyConnected, self).__init__(
            layer_sizes=[784, 20, 10, 10, 10]
        )
        
    def get_name(self):
        return "SmallFullyConnected" 