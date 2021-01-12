# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import time


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, root=None, name=None, device='cuda'):
        if root is None:
            root = './checkpoints/'
        if root[-1] != '/':
            root = root + '/'
        if name is None:
            prefix = self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        if device == 'cuda':
            torch.save(self.module.state_dict(), root + name)
        elif device == 'cpu':
            torch.save(self.state_dict(), root + name)
        return name
