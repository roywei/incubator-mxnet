import mxnet as mx
from mxnet import gluon
from mxnet.gluon import estimator, nn, data as gdata
from mxnet.gluon.estimator import estimator
import os
import sys
net = nn.Sequential()

net.add(nn.Conv2D(32, kernel_size=3, activation='relu'),
        nn.Conv2D(64, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2),
        nn.Dropout(0.25),
        nn.Flatten(),
        nn.Dense(128, activation="relu"), nn.Dropout(0.5),
        nn.Dropout(0.5),
        nn.Dense(10))


def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join(
        '~', '.mxnet', 'datasets', 'fashion-mnist')):
    root = os.path.expanduser(root)  # Expand the user path '~'.
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)
    mnist_train = gdata.vision.MNIST(root=root, train=True)
    mnist_test = gdata.vision.MNIST(root=root, train=False)
    num_workers = 0 if sys.platform.startswith('win32') else 4
    train_iter = gdata.DataLoader(
        mnist_train.transform_first(transformer), batch_size, shuffle=True,
        num_workers=num_workers)
    test_iter = gdata.DataLoader(
        mnist_test.transform_first(transformer), batch_size, shuffle=False,
        num_workers=num_workers)
    return train_iter, test_iter

batch_size = 128
train_data, test_data = load_data_fashion_mnist(batch_size, resize=28)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
acc = mx.metric.Accuracy()
net.initialize()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
est= estimator.Estimator(net=net, loss=loss, metrics=acc, trainers=trainer)
est.fit(train_data=train_data, epochs=2)
