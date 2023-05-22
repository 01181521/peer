import torch
import torch.nn as nn
from torchvision.models import alexnet

from models import register_network
from pyramid_vig import *
import torch.nn as nn
import torch.nn.functional as F


class CosSim(nn.Module):
    def __init__(self, nfeat, nclass, codebook=None, learn_cent=True):
        super(CosSim, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.learn_cent = learn_cent

        if codebook is None:  # if no centroids, by default just usual weight
            codebook = torch.randn(nclass, nfeat)

        self.centroids = nn.Parameter(codebook.clone())
        if not learn_cent:
            self.centroids.requires_grad_(False)

    def forward(self, x):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(x, norms)

        norms_c = torch.norm(self.centroids, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centroids, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        return logits

    def extra_repr(self) -> str:
        return 'in_features={}, n_class={}, learn_centroid={}'.format(
            self.nfeat, self.nclass, self.learn_cent
        )


@register_network('alexnet')
class AlexNet(nn.Module):
    def __init__(self,
                 nbit, nclass, pretrained=False, freeze_weight=False,
                 codebook=None,
                 **kwargs):
        super(AlexNet, self).__init__()

        # model = alexnet(pretrained=pretrained)
        self.model = pvig_b_224_gelu()
       
        state_dict = torch.load('/hdd/sxz/0code/vig/pvig_s_224_gelu.pth')
        # model.hash =  nn.Conv2d(1024, nbit, kernel_size=(1, 1), stride=(1, 1))

        # self.model = pvig_s_224_gelu()
        # # self.model.prediction[4] =  nn.Conv2d(1024, nbit, kernel_size=(1, 1), stride=(1, 1))
        # state_dict = torch.load('/hdd/sxz/0code/vig_pytorch/path/to/pretrained/model/pvig_s_82.1.pth.tar')
        # # state_dict = torch.load('/root/code/pvig_s_82.1.pth.tar')
        
        # model.load_state_dict(state_dict, strict=False)
        print (self.model.prediction)
        pre_dict = {k: v for k, v in state_dict.items() if "prediction.4" not in k}
        self.model.load_state_dict(pre_dict, strict=False)
        # self.features = model.stem
        # self.avgpool = model.backbone
        # fc = []
        # for i in range(4):
        #     fc.append(model.prediction[i])
        # self.fc = nn.Sequential(*fc)

        # in_features = model.prediction[4].in_features
        if codebook is None:  # usual CE
            self.ce_fc = nn.Linear(nbit, nclass)
        else:
            # not learning cent, we are doing codebook learning
            self.ce_fc = CosSim(nbit, nclass, codebook, learn_cent=False)

        self.hash_fc = nn.Sequential(
            nn.Conv2d(1024, nbit, kernel_size=(1, 1), stride=(1, 1)),
            # nn.BatchNorm2d(nbit, momentum=0.1)
            # nn.Linear(1024, nbit, bias=False),
            # nn.BatchNorm1d(nbit, momentum=0.1)
        )

        # self.hash_fc = nn.Sequential(
        #     nn.Conv2d(1024, nbit, kernel_size=(1, 1), stride=(1, 1)),
        #     # nn.BatchNorm2d(nbit, momentum=0.1)
        #     # nn.Linear(1024, nbit, bias=False),
        #     # nn.BatchNorm1d(nbit, momentum=0.1)
        # )

        # nn.init.normal_(self.hash_fc[0].weight, std=0.01)
        # nn.init.zeros_(self.hash_fc.bias)

        self.extrabit = 0

        if freeze_weight:
            for param in self.features.parameters():
                param.requires_grad_(False)
            for param in self.fc.parameters():
                param.requires_grad_(False)

    def get_backbone_params(self):
        return list(self.model.parameters()) 

    def get_hash_params(self):
        return list(self.ce_fc.parameters()) + list(self.hash_fc.parameters())

    # def get_backbone_params(self):
    #     return list(self.features.parameters()) + list(self.fc.parameters())

    # def get_hash_params(self):
    #     return list(self.ce_fc.parameters()) + list(self.hash_fc.parameters())
    
    def forward(self, x):
        x = self.model(x)
        v = self.hash_fc(x).squeeze(-1).squeeze(-1)
        u = self.ce_fc(v)
        return u, v

    # def forward(self, x):
    #     x = self.features(x)
    #     x = self.avgpool(x)
    #     # x = torch.flatten(x, 1)
    #     x = self.fc(x)
    #     v = self.hash_fc(x)
    #     v = F.adaptive_avg_pool2d(v, 1)
    #     v = v.squeeze(-1).squeeze(-1)
    #     u = self.ce_fc(v)
    #     return u, v

# class AlexNet(nn.Module):
#     def __init__(self,
#                  nbit, nclass, pretrained=False, freeze_weight=False,
#                  codebook=None,
#                  **kwargs):
#         super(AlexNet, self).__init__()

#         model = alexnet(pretrained=pretrained)
#         self.features = model.features
#         self.avgpool = model.avgpool
#         fc = []
#         for i in range(6):
#             fc.append(model.classifier[i])
#         self.fc = nn.Sequential(*fc)

#         in_features = model.classifier[6].in_features
#         if codebook is None:  # usual CE
#             self.ce_fc = nn.Linear(nbit, nclass)
#         else:
#             # not learning cent, we are doing codebook learning
#             self.ce_fc = CosSim(nbit, nclass, codebook, learn_cent=False)

#         self.hash_fc = nn.Sequential(
#             nn.Linear(in_features, nbit, bias=False),
#             nn.BatchNorm1d(nbit, momentum=0.1)
#         )

#         nn.init.normal_(self.hash_fc[0].weight, std=0.01)
#         # nn.init.zeros_(self.hash_fc.bias)

#         self.extrabit = 0

#         if freeze_weight:
#             for param in self.features.parameters():
#                 param.requires_grad_(False)
#             for param in self.fc.parameters():
#                 param.requires_grad_(False)

#     def get_backbone_params(self):
#         return list(self.features.parameters()) + list(self.fc.parameters())

#     def get_hash_params(self):
#         return list(self.ce_fc.parameters()) + list(self.hash_fc.parameters())

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         v = self.hash_fc(x)
#         u = self.ce_fc(v)
#         return u, v
