# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt

class Model(torch.nn.Module):
    def __init__(self, name):
        super(Model, self).__init__()
        func = torch.nn.Sequential()

        if name == 'sigmoid':
            func.add_module('sigmoid', torch.nn.Sigmoid())
        elif name == 'tanh':
            func.add_module('tanh', torch.nn.Tanh())
        elif name == 'relu':
            func.add_module('relu', torch.nn.ReLU())
        elif name == 'selu':
            func.add_module('selu', torch.nn.SELU())
        elif name == 'softplus':
            func.add_module('softplus', torch.nn.Softplus())
        elif name == 'leakyReLU':
            func.add_module('leakyReLU', torch.nn.LeakyReLU(negative_slope=0.05))
        elif name == 'elu':
            func.add_module('elu', torch.nn.ELU())

        self.func = func

    def forward(self, x):
        return self.func(x)

model_sigmoid = Model('sigmoid')
model_tanh = Model('tanh')
model_relu = Model('relu')
model_softplus = Model('softplus')
model_leakyReLU = Model('leakyReLU')
model_selu = Model('selu')
model_elu = Model('elu')

x = torch.autograd.Variable(torch.arange(start=-3, end=3, step=0.01, out=None))
y_sigmoid = model_sigmoid(x)
y_tanh = model_tanh(x)
y_relu = model_relu(x)
y_selu = model_selu(x)
y_elu = model_elu(x)
y_softplus = model_softplus(x)
y_leakyReLU = model_leakyReLU(x)


x = x.data.numpy()
y_sigmoid = y_sigmoid.data.numpy()
y_tanh = y_tanh.data.numpy()
y_relu = y_relu.data.numpy()
y_selu = y_selu.data.numpy()
y_elu = y_elu.data.numpy()
y_softplus = y_softplus.data.numpy()
y_leakyReLU = y_leakyReLU.data.numpy()


fig = plt.figure()
ax = fig.add_subplot(111)

sigmoid, = ax.plot(x,y_sigmoid, label=u'sigmoid')
tanh, = ax.plot(x,y_tanh, label=u'tanh')
softplus, = ax.plot(x,y_softplus, label=u'Softplus')
relu, = ax.plot(x,y_relu, label=u'ReLU')
leakyReLU, = ax.plot(x,y_leakyReLU, label=u'LeakyReLU')
elu, = ax.plot(x,y_elu, label=u'ELU')
selu, = ax.plot(x,y_selu, label=u'SELU')

plt.axis([-3.0,3.0,-2.0,2.0])
plt.grid(True, linestyle = "-.")
plt.xlabel("x")  # X轴标签
plt.ylabel("f(x)")  # Y轴标签
plt.title("Activation function")  # 标题
plt.legend(loc='upper left')
plt.show()
