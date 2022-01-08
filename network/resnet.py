import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torchvision
import kornia as K


# Define affine transformation
def get_affine_transform(return_transform=False):
    affine_transform = K.augmentation.RandomAffine(degrees=(-15, 15),translate=(0, 0.2), scale=(0.9, 1.1), return_transform=return_transform, p=1.0)
    #affine_transform = K.augmentation.RandomAffine(degrees=(-15, 15),translate=(0, 0.1), scale=(0.95, 1.05), return_transform=return_transform, p=1.0)
    return affine_transform


class ResNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        
        self.model = torchvision.models.resnet152(pretrained) 
        self.in_channel = self.model.fc.in_features
        self.class_aware_fc = nn.Conv2d(self.in_channel, num_classes, kernel_size=1, bias=False)
    
    def forward(self, x):
        # Record input
        input_x = x

        # Feeding input into convolutional embedding part
        for name, module in self.model._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        # Set feature
        feature = x
        _,_,h,w = feature.size()
        x_s = F.interpolate(input_x,(h,w),mode='bilinear',align_corners=False)
        feature = torch.cat([x_s, feature], dim=1)
        
        # Pass feature to class-aware fully convolutional layer
        cam = self.class_aware_fc(x)

        # Apply global avarage pooling
        score = F.adaptive_avg_pool2d(cam, (1, 1))
        
        # Flattening
        score = torch.flatten(score, 1)
        
        return feature, cam, score

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


class ResNetUp(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        
        self.model = torchvision.models.resnet18(pretrained) 
        self.in_channel = self.model.fc.in_features
        self.up_conv1 = nn.ConvTranspose2d(in_channels=self.in_channel, out_channels=self.in_channel//2, kernel_size=(4, 4),
                                          stride=2, padding=1, output_padding=0, bias=False)                                 
        self.class_aware_fc = nn.Conv2d(self.in_channel//2, num_classes, kernel_size=1, bias=False)
    
    def forward(self, x):
        # Record input
        input_x = x

        # Feeding input into convolutional embedding part
        for name, module in self.model._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        # Up convolution
        x = self.up_conv1(x)

        # Set feature
        feature = x
        _,_,h,w = feature.size()
        x_s = F.interpolate(input_x,(h,w),mode='bilinear',align_corners=False)
        feature = torch.cat([x_s, feature], dim=1)
        
        # Pass feature to class-aware fully convolutional layer
        cam = self.class_aware_fc(x)

        # Apply global avarage pooling
        score = F.adaptive_avg_pool2d(cam, (1, 1))
        
        # Flattening
        score = torch.flatten(score, 1)
        
        return feature, cam, score

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


class ResNet_ER_PCM(ResNet):  
    def __init__(self, num_classes, pretrained=True):
        super().__init__(num_classes, pretrained)
        self.feature_embedding = nn.Conv2d(self.in_channel+3, 256, kernel_size=1, bias=False)
        self.affine_matrix = None
    
    def get_affine_matrix(self):
        return self.affine_matrix

    def PCM(self, cam, f):
        n,c,h,w = f.size()
        f = self.feature_embedding(f)
        f = f.view(n,-1,h*w)
        f = f/(torch.norm(f,dim=1,keepdim=True)+1e-5)
        
        aff = F.relu(torch.matmul(f.transpose(1,2), f),inplace=True)
        aff = aff/(torch.sum(aff,dim=1,keepdim=True)+1e-5)
        
        cam = F.interpolate(cam, (h,w), mode='bilinear', align_corners=False).view(n,-1,h*w)
        cam_rv = torch.matmul(cam, aff).view(n,-1,h,w)
        
        del f, aff, cam

        return cam_rv
    
    def norm_CAM(self, cam):
        n,c,h,w = cam.size()
        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n,c,-1), dim=-1)[0].view(n,c,1,1)+1e-5
            cam_d_norm = F.relu(cam_d-1e-5)/cam_d_max
            cam_d_norm[:,0,:,:] = 1-torch.max(cam_d_norm[:,1:,:,:], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:,1:,:,:], dim=1, keepdim=True)[0]
            cam_d_norm[:,1:,:,:][cam_d_norm[:,1:,:,:] < cam_max] = 0
        
        del cam, cam_d, cam_d_max, cam_max

        return cam_d_norm

    def net_forward(self, input_img, mode='train'):
        # Feed input images through network
        _, _, H, W = input_img.size() 
        f, cam, score = self.forward(input_img)
        
        # Get CAM and CAM revised
        cam_d_norm = self.norm_CAM(cam)
        cam_rv = F.interpolate(self.PCM(cam_d_norm, f), (H,W), mode='bilinear', align_corners=False)
        #print(H, W)
        cam = F.interpolate(cam_d_norm, (H,W), mode='bilinear', align_corners=False)

        if mode == 'val':
            return cam_rv
        if mode == 'test':
            return cam_rv, score

        # Scale to get affine images
        scale_factor = 0.6
        input_img2 = F.interpolate(input_img,scale_factor=scale_factor,mode='bilinear',align_corners=False)
        _, _, H, W = input_img2.size() 
        
        # Feed affine transformation of input images through network
        affine, matrix = get_affine_transform(return_transform=True)(input_img2)
        self.affine_matrix = matrix
        affine_f, affine_cam, affine_score = self.forward(affine)
        
        # Get affine CAM and affine CAM revised
        affine_cam_d_norm = self.norm_CAM(affine_cam)
        affine_cam_rv = F.interpolate(self.PCM(affine_cam_d_norm, affine_f), (H,W), mode='bilinear', align_corners=False)
        affine_cam = F.interpolate(affine_cam_d_norm, (H,W), mode='bilinear', align_corners=False)
        
        del f, cam_d_norm, input_img2, affine, matrix, affine_f, affine_cam_d_norm

        return (score, cam, cam_rv), (affine_score, affine_cam, affine_cam_rv)
    

    
        
