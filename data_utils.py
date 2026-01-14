import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Class labels for Fashion-MNIST
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def load_fashion_mnist(num_samples=10000):
    """
    Load Fashion-MNIST dataset.
    Optionally limit the number of training samples.
    """
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train[:num_samples]
    y_train = y_train[:num_samples]
    x_test = x_test[:int(num_samples/2)]
    y_test =y_test[:int(num_samples/2)]

    return x_train, y_train, x_test, y_test


def preprocess_data(x):
    """
    Flatten and normalize images.
    Input shape: (N, 28, 28)
    Output shape: (N, 784)
    """
    x = x.reshape(x.shape[0], -1)   # flatten
    x = x.astype(np.float32) / 255.0
    return x


def plot_one_sample_per_class(x, y):
    """
    Plot one image per Fashion-MNIST class.
    """
    plt.figure(figsize=(10, 5))
    shown = set()
    idx = 0
    plot_idx = 1

    while len(shown) < 10:
        label = y[idx]
        if label not in shown:
            plt.subplot(2, 5, plot_idx)
            plt.imshow(x[idx], cmap="gray")
            plt.title(CLASS_NAMES[label])
            plt.axis("off")

            shown.add(label)
            plot_idx += 1
        idx += 1

    plt.tight_layout()
    plt.show()


