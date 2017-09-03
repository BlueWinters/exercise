
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, sqrt


def plot_gaussian_mixture():
    def sample(x, y, label, n_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    n_dim = 2
    n_labels = 10
    x_var = 0.5
    y_var = 0.1
    batch_size = 1000

    x = np.random.normal(0, x_var, [batch_size, 1])
    y = np.random.normal(0, y_var, [batch_size, 1])
    z = np.empty([batch_size, n_dim], dtype=np.float32)
    for batch in range(batch_size):
        z[batch, 0:2] = sample(x[batch], y[batch], np.random.randint(0, n_labels), n_labels)
    plt.scatter(z[:,0], z[:,1], edgecolors='face')
    plt.show()




if __name__ == '__main__':
    gaussian_mixture()
