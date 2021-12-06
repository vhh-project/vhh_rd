import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

class FeatureExtractor(nn.Module):
    def __init__(self, model_name, evaluate = True):
        super(FeatureExtractor, self).__init__()
        self.model_name = model_name
        if model_name == "resnet152":
            resnet152 = models.resnet152(pretrained=True)
            modules = list(resnet152.children())[:-1]
            resnet152 = nn.Sequential(*modules)
            self.model = resnet152
        elif model_name == "resnet50":
            resnet50 = models.resnet50(pretrained=True)
            modules = list(resnet50.children())[:-1]
            resnet152 = nn.Sequential(*modules)
            self.model = resnet50
        else:
            print("Unknown model")
            quit()

        if evaluate:
            print("Evaluation mode")
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def get_preprocessing(self, siamese = False):
        """
        If siamese is true then the image will not be turned to a tensor to allow augmentations
        """
        if self.model_name == "resnet152" or self.model_name == "resnet50":
            return transforms.Compose([
                transforms.ToTensor() if not siamese else torch.nn.Identity(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    def forward(self, x):
        return self.model(x) 