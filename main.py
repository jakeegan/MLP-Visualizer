import numpy as np
from flask import Flask, render_template

app = Flask(__name__)

# Threshold Logic Units
W_OR = np.array([[[-0.5, 1, 1]]])
W_NAND = np.array([[[1.5, -1, -1]]])
W_AND = np.array([[[-1.5, 1, 1]]])
W_XOR = [[[-0.5, 1, 1], [1.5, -1, -1]], [[-1.5, 1, 1]]]


class Network:
    """
    Holds data for a multi-layer perceptron network
    """

    def __init__(self, weights):
        self.weights = weights

    @staticmethod
    def _step_function(a):
        """
        Applies Heaviside step function to the activation
        """
        if a > 0:
            return 1
        else:
            return 0

    @staticmethod
    def _calc_activation(inputs, weights):
        """
        Calculate total activation for the perceptron
        """
        return np.dot(inputs, weights)

    def feed_forward(self, input):
        """
        Calculate outputs at each layer
        """
        layer0 = input
        layer0.insert(0, 1)
        layers = [layer0]

        for i in range(len(self.weights)):   # Loop through layers
            temp = []
            for j in range(len(self.weights[i])):    # Loop through neurons
                temp.append(self._step_function(self._calc_activation(layers[i], self.weights[i][j])))
            temp.insert(0, 1)
            layers.append(temp)
        # Remove bias
        for i in range(len(layers)):
            del layers[i][0]
        return layers


@app.route('/')
def index():
    return render_template("index.html", net=net, weights=weights, net_width=net_width, net_height=net_height)


if __name__ == '__main__':
    net_width = 3
    net_height = 2
    weights = W_XOR
    network = Network(weights)
    net = network.feed_forward([1, 0])
    app.run()
