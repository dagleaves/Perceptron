import numpy as np

class Perceptron:

    def __init__(self):
        self.weights_shape = [None, None]
        self.weights = [np.random.standard_normal(
            w) for w in self.weights_shape]
        self.lr = 2**.5

    @staticmethod
    def activate(a: float):
        return 1 if a >= 0.0 else 0

    def predict(self, inputs: list[float]):
        sum = 0.0
        for i in range(len(self.weights)):
            sum += self.weights[i] * inputs[i]
        return self.activate(sum)

    def train(self, inputs: list[float], target: int):
        prediction = self.predict(inputs)
        error = target - prediction
        # print(error)
        for i in range(len(self.weights)):
            self.weights[i] += error * inputs[i] * self.lr
        # print(self.weights)
