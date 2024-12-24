
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def get_model(args):
    if args.model == "ResNet18":
        return torchvision.models.resnet18(
            num_classes=args.num_classes, 
            pretrained=args.pretrained, 
        )
    elif args.model == "ResNet34":
        return torchvision.models.resnet34(
            num_classes=args.num_classes, 
            pretrained=args.pretrained, 
        )
    else:
        raise NotImplementedError

# Define customized local model
