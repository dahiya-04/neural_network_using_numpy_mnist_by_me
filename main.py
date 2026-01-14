from data_utils import load_fashion_mnist, preprocess_data
from model import NeuralNetwork
from optimizer import Adam,SGD , Nestrov , RMSProp
from train import train

# Load data
x_train, y_train, x_test, y_test = load_fashion_mnist(num_samples=1000)
x_train = preprocess_data(x_train)

# Model
model = NeuralNetwork([784, 128, 64, 10])

# Choose optimizer (just change this line)
optimizer = Adam(lr=0.001)
# optimizer = SGD(lr=0.01)
# optimizer = MSGD(lr=0.01)
# optimizer = RMSProp(lr=0.001)


train(model, optimizer, x_train, y_train,
      epochs=10, batch_size=32)
