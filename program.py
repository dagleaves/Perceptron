from perceptron import Perceptron
import matplotlib.pyplot as plt
import numpy as np

SIZE = 400
TRAIN_MULT = 1

train_x, train_y = SIZE * np.random.rand(2, SIZE * TRAIN_MULT)
test_x, test_y = SIZE * np.random.rand(2, SIZE)
labels = [1 if b >= a else 0 for a, b in zip(train_x, train_y)]
nand = Perceptron()

# Train perceptron
for i, (a, b) in enumerate(zip(train_x, train_y)):
    print(i)
    nand.train([a, b], labels[i])


# Test perceptron
colors = []
for i, (a, b) in enumerate(zip(train_x, train_y)):
    prediction = nand.predict([a, b])
    target = labels[i]
    if prediction == target:
        colors.append('tab:green')
    else:
        colors.append('tab:red')

# Display points
plt.scatter(train_x, train_y, c=colors, edgecolors='k', alpha=1)
plt.plot([0, SIZE], [0, SIZE])
plt.show()
