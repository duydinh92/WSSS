import os
import torch
import random
import numpy as np
import torch.nn.functional as F


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


def max_norm(p, e=1e-5):
    N, C, H, W = p.size()
    p = F.relu(p)
    max_v = torch.max(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
    min_v = torch.min(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
    p = F.relu(p-min_v-e)/(max_v-min_v+e)

    return p


def adaptive_min_pooling_loss(x):
    n,c,h,w = x.size()
    k = int(0.2*h*w)
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n,-1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y)/(k*n)
    
    return loss


def max_onehot(x):
    n,c,h,w = x.size()
    x_max = torch.max(x[:,:,:,:], dim=1, keepdim=True)[0]
    x[:,:,:,:][x[:,:,:,:] != x_max] = 0
    
    return x


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