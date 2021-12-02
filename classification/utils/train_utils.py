import os
import torch
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))


def save_model(model, model_path):
  torch.save(model.state_dict(), model_path)


def get_learning_rate_from_optimizer(optimizer):
    return optimizer.param_groups[0]['lr']


def get_numpy_from_tensor(tensor):
    return tensor.cpu().detach().numpy()


class PolyOptimizer(torch.optim.SGD):
    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9, nesterov=False):
        super().__init__(params, lr, weight_decay, nesterov=nesterov)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum
        
        self.__initial_lr = [group['lr'] for group in self.param_groups]
    
    def step(self, closure=None):
        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1