import torch
import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        if model == "resnet152":
            resnet152 = models.resnet152(pretrained=True)
            modules = list(resnet152.children())[:-1]
            resnet152 = nn.Sequential(*modules)
            for p in resnet152.parameters():
                p.requires_grad = False

            self.model = resnet152
        else:
            print("Unknown model")
            quit()

    def forward(self, x):
        return self.model(x) 