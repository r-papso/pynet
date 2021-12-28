# Pynet

Pynet is a Python library containing all of the deep learning building blocks.

## Features

### _Tensor operations_

Tensor operations are included in the [pynet.functional](./pynet/functional) namespace. Pynet provides a couple of basic tensor operations such as matrix multiplication or tensor addition. New tensor operation can be created by implementing the [Function](./pynet/functional/abstract.py) abstract class. Tensor functions that are included in the library are:

- Tensor addition [[link]](./pynet/functional/add.py)
- Matrix multiplication [[link]](./pynet/functional/matmul.py)
- Element-wise maximum [[link]](./pynet/functional/max.py)
- Element-wise sigmoid [[link]](./pynet/functional/sigmoid.py)

### _Neural network layers and activation functions_

Implementation of neural network layers and activation functions are included in the [pynet.nn](./pynet/nn) namespace. The namespace contains basic layer and activation function implementations and can be extended by implementing the [Module](./pynet/nn/abstract.py) abstract class. List of layers and activation functions included in the Pynet library:

- Fully connected layer (Linear) [[link]](./pynet/nn/linear.py)
- Sequential module (container for other modules) [[link]](./pynet/nn/sequential.py)
- Rectified linear unit (ReLU) activation function [[link]](./pynet/nn/relu.py)
- Sigmoid activation function [[link]](./pynet/nn/sigmoid.py)

### _Loss functions_

Loss function implementations are included in the [pynet.loss](./pynet/loss) namespace. Custom loss functions can be created by implementing the [Loss](./pynet/loss/abstract.py) abstract class. Loss functions included in the Pynet library are:

- Binary cross entropy [[link]](./pynet/loss/bce.py)
- Mean squared error [[link]](./pynet/loss/mse.py)

### _Optimization algorithms_

Optimization algorithm implementations are included in the [pynet.optimizers](./pynet/optimizers) namespace. New optimization algorithm can be created by implementing the [Optimizer](./pynet/optimizers/abstract.py) abstract class. Optimization algorithms included in the library:

- Stochastic Gradient Descent (SGD) [[link]](./pynet/optimizers/sgd.py)

### _Data manipulation_

[pynet.data](./pynet/data) namespace provides abstraction over the datasets used for training the neural network. [In-memory](./pynet/data/in_memory.py) dataset implementation is already included in the library, however one can create their own dataset implementation by extending the [Dataset](./pynet/data/abstract.py) abstract class.

### _Neural network training_

For the neural network's training/testing procedure, one can either use default implementation of these procedures, which can be found in the [DefaultTrainer](./pynet/training/trainer/default.py) class or implement their own training/testing logic by extending the [Trainer](./pynet/training/trainer/abstract.py) abstract class. One can also omit both these options and implement its own logic from scratch (for example, see the [DefaultTrainer](./pynet/training/trainer/default.py) train and test function implementations).

When using one of the [Trainer](./pynet/training/trainer/abstract.py)'s implementations, a [Callback](./pynet/training/callbacks/abstract.py) class was created to perform additional actions at various stages of training/testing procedure. There are a couple of callbacks implemented and ready to use, such as:

- Callback providing printing all the measured model's metrics onto the console ([PrintCallback](./pynet/training/callbacks/print.py))
- Callback providing optimizer's learning rate scheduling ([LrSchedule](./pynet/training/callbacks/lr_schedule.py))

One can also easily create their own callback by extending the [Callback](./pynet/training/callbacks/abstract.py) abstract class.

### _Summary_

Table below summarizes the feature section. Every row of a table contains name of the namespace, namespace description (what functionality this namespace provides) and a way of extending this functionality (by implementing the abstract class).

| Namespace                                              | Functionality                                     | Abstract class                                     |
| ------------------------------------------------------ | ------------------------------------------------- | -------------------------------------------------- |
| [pynet.functional](./pynet/functional)                 | Tensor operations                                 | [Function](./pynet/functional/abstract.py)         |
| [pynet.nn](./pynet/nn)                                 | Neural network layers and activation functions    | [Module](./pynet/nn/abstract.py)                   |
| [pynet.loss](./pynet/loss)                             | Loss functions                                    | [Loss](./pynet/loss/abstract.py)                   |
| [pynet.optimizers](./pynet/optimizers)                 | Optimization algorithms                           | [Optimizer](./pynet/optimizers/abstract.py)        |
| [pynet.data](./pynet/data)                             | Data manipulation                                 | [Dataset](./pynet/data/abstract.py)                |
| [pynet.training.trainer](./pynet/training/trainer)     | Neural network training/testing procedure         | [Trainer](./pynet/training/trainer/abstract.py)    |
| [pynet.training.callbacks](./pynet/training/callbacks) | Callback functions during the training/testing    | [Callback](./pynet/training/callbacks/abstract.py) |

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pynet.

```bash
pip install pynet-dl
```

## Usage

Here is the simple example how to use the Pynet library. In this example we are going to train a very small neural network on the make_circles dataset from sklearn.dataset package.

### _Training_

First, we need to import all the necessary stuff:

```python
import numpy as np

from sklearn.datasets import make_circles

# Neural network modules
from pynet.nn.sequential import Sequential
from pynet.nn.linear import Linear
from pynet.nn.relu import ReLU
from pynet.nn.sigmoid import Sigmoid

# Datasets
from pynet.data.in_memory import InMemoryDataset

# Loss functions
from pynet.loss.bce import BinaryCrossEntropy

# Optimizers
from pynet.optimizers.sgd import SGD

# Weight initializers
from pynet.initializers.he_normal import HeNormal

# Trainer and training/testing callbacks
from pynet.training.trainer.default import DefaultTrainer
from pynet.training.callbacks.print import PrintCallback
```

Then, load the dataset and do preprocessing. As the single input to the neural network (i. e. sample xi) is supposed to be of shape [n_features x 1] (i. e. column vector), we need to add one extra dimension to the X array:

```python
X, y = make_circles(n_samples=1000, noise=0.025)
# inputs to neural net must be of shape [n, 1]
X = np.expand_dims(X, axis=2)
```

Then, we will create the network, dataset, optimizer, appropriate loss function, trainer and optionally also specify a list of callbacks:

```python
epochs = 20

model = Sequential([
    Linear(inputs=2, neurons=16, initializer=HeNormal()),
    ReLU(),
    Linear(inputs=16, neurons=1, initializer=HeNormal()),
    Sigmoid()
])

dataset = InMemoryDataset(X, y)
loss_f = BinaryCrossEntropy()
sgd = SGD(learning_rate=0.01, momentum=0.9)
callbacks = [PrintCallback(), LrSchedule(optimizer=sgd, schedule=lr_schedule)]
trainer = DefaultTrainer()
```

And then run the training:

```python
trainer.train(
    model=model,
    train_dataset=dataset,
    val_dataset=None,
    loss_f=loss_f,
    optimizer=sgd,
    epochs=epochs,
    callbacks=callbacks
)
```

### _Inference_

After training the neural network, we can use it for inference.

```python
# Make prediction for the i-th sample in the dataset
i = 0
y_pred = model.forward(X[i]).ndarray.item()
```

## Examples

More examples of the Pynet library usage can be found in the [notebooks](./notebooks) directory.

## License

[MIT](https://choosealicense.com/licenses/mit/)
