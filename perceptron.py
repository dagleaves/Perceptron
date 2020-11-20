import matplotlib.pyplot as plt
import numpy as np

SIZE = 400
TRAIN_MULT = 1


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
        for i in range(len(self.weights)):
            self.weights[i] += error * inputs[i] * self.lr


train_x, train_y = SIZE * np.random.rand(2, SIZE * TRAIN_MULT)
train_labels = [1 if b >= a else 0 for a, b in zip(train_x, train_y)]
test_x, test_y = SIZE * np.random.rand(2, SIZE)
test_labels = [1 if b >= a else 0 for a, b in zip(test_x, test_y)]
nand = Perceptron()

# Train perceptron
for i, (a, b) in enumerate(zip(train_x, train_y)):
    nand.train([a, b], train_labels[i])


# Test perceptron
colors = []
for i, (a, b) in enumerate(zip(test_x, test_y)):
    prediction = nand.predict([a, b])
    target = test_labels[i]
    if prediction == target:
        colors.append('tab:green')
    else:
        colors.append('tab:red')

# Display points
plt.scatter(test_x, test_y, c=colors, edgecolors='k', alpha=1)
plt.plot([0, SIZE], [0, SIZE])
plt.show()
