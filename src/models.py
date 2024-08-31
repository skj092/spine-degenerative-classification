import torch.nn as nn
import timm
import torch


class RSNA24Model(nn.Module):
    def __init__(self, model_name, in_c=20, n_classes=75, pretrained=True, features_only=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=in_c,
                                       num_classes=n_classes, features_only=features_only, global_pool='avg')

    def forward(self, xb):
        return self.model(xb)



if __name__ == '__main__':
    model = RSNA24Model('resnet18', in_c=20, n_classes=75, pretrained=True, features_only=False)
    print(model)

