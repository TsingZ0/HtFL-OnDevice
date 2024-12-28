
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from flwr.common.logger import log
from logging import WARNING, INFO


def save_item(item, item_name, item_path=None):
    if not os.path.exists(item_path):
        os.makedirs(item_path)
    torch.save(item, os.path.join(item_path, item_name + ".pt"))

def load_item(item_name, item_path=None):
    try:
        return torch.load(os.path.join(item_path, item_name + ".pt"))
    except FileNotFoundError:
        log(INFO, f'{item_name} Not Found')
        return None
    

# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, args, model):
        super().__init__()

        self.base = model
        if hasattr(self.base, 'heads'):
            self.base.heads = nn.AdaptiveAvgPool1d(args.feature_dim)
        elif hasattr(self.base, 'head'):
            self.base.head = nn.AdaptiveAvgPool1d(args.feature_dim)
        elif hasattr(self.base, 'fc'):
            self.base.fc = nn.AdaptiveAvgPool1d(args.feature_dim)
        elif hasattr(self.base, 'classifier'):
            self.base.classifier = nn.AdaptiveAvgPool1d(args.feature_dim)
        else:
            raise('The base model does not have a classification head.')

        self.head = nn.Linear(args.feature_dim, args.num_classes)
        
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out
    

def get_model(args):
    if args.model == "ResNet18":
        model = torchvision.models.resnet18(
            num_classes=args.num_classes, 
            pretrained=args.pretrained, 
        )
    elif args.model == "ResNet34":
        model = torchvision.models.resnet34(
            num_classes=args.num_classes, 
            pretrained=args.pretrained, 
        )
    else:
        raise NotImplementedError    
    return BaseHeadSplit(args, model)


def get_auxiliary_model(args):
    if args.auxiliary_model == "ResNet18":
        model = torchvision.models.resnet18(
            num_classes=args.num_classes, 
            pretrained=args.pretrained, 
        )
    elif args.auxiliary_model == "ResNet34":
        model = torchvision.models.resnet34(
            num_classes=args.num_classes, 
            pretrained=args.pretrained, 
        )
    else:
        raise NotImplementedError    
    return BaseHeadSplit(args, model)


# Define customized local model

