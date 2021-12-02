import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torchvision


class DenseNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        
        self.model = torchvision.models.densenet121(pretrained) 
        self.model_embedding = self.model.features
        self.in_channel = self.model.classifier.in_features
        self.class_aware_fc = nn.Conv2d(self.in_channel, num_classes, kernel_size=1, bias=False)
    
    def forward(self, x):
        # Feeding input into convolutional embedding part
        for name, module in self.model_embedding._modules.items():
            if name == 'norm5':
                break
            x = module(x)

        # Relu    
        feature = F.relu(x, inplace=False)
        
        # Pass feature to class-aware fully convolutional layer
        cam = self.class_aware_fc(feature)

        # Apply global avarage pooling
        score = F.adaptive_avg_pool2d(cam, (1, 1))
        
        # Flattening
        score = torch.flatten(score, 1)
        
        return score, cam

    def get_parameter_groups(self, print_fn=print):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():
            # pretrained weights
            if 'model' in name:
                if 'weight' in name:
                    if print_fn is not None:
                        print_fn(f'pretrained weights : {name}')
                    groups[0].append(value)
                else:
                    if print_fn is not None:
                        print_fn(f'pretrained bias : {name}')
                    groups[1].append(value)
                    
            # scracthed weights
            else:
                if 'weight' in name:
                    if print_fn is not None:
                        print_fn(f'scratched weights : {name}')
                    groups[2].append(value)
                else:
                    if print_fn is not None:
                        print_fn(f'scratched bias : {name}')
                    groups[3].append(value)
        return groups


class DenseNetUp(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        
        self.model = torchvision.models.densenet121(pretrained) 
        self.model_embedding = self.model.features
        self.in_channel = self.model.classifier.in_features
        self.up_conv1 = nn.ConvTranspose2d(in_channels=self.in_channel, out_channels=self.in_channel//4, kernel_size=(4, 4),
                                          stride=2, padding=1, output_padding=0, bias=False) 
        self.class_aware_fc = nn.Conv2d(self.in_channel//4, num_classes, kernel_size=1, bias=False)
    
    def forward(self, x):
        # Feeding input into convolutional embedding part
        for name, module in self.model_embedding._modules.items():
            if name == 'norm5':
                break
            x = module(x)
        
        # Up convolution
        x = self.up_conv1(x)

        # Relu    
        feature = F.relu(x, inplace=False)
        
        # Pass feature to class-aware fully convolutional layer
        cam = self.class_aware_fc(feature)

        # Apply global avarage pooling
        score = F.adaptive_avg_pool2d(cam, (1, 1))
        
        # Flattening
        score = torch.flatten(score, 1)
        
        return score, cam

    def get_parameter_groups(self, print_fn=print):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():
            # pretrained weights
            if 'model' in name:
                if 'weight' in name:
                    if print_fn is not None:
                        print_fn(f'pretrained weights : {name}')
                    groups[0].append(value)
                else:
                    if print_fn is not None:
                        print_fn(f'pretrained bias : {name}')
                    groups[1].append(value)
                    
            # scracthed weights
            else:
                if 'weight' in name:
                    if print_fn is not None:
                        print_fn(f'scratched weights : {name}')
                    groups[2].append(value)
                else:
                    if print_fn is not None:
                        print_fn(f'scratched bias : {name}')
                    groups[3].append(value)
        return groups