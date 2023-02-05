import os
import torch
import time
import numpy as np
import json
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision import models
import pickle
from web3 import Web3
from contract import *

torch.multiprocessing.set_sharing_strategy('file_system')


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}
class ResNet(nn.Module):
    def __init__(self, hash_bit, res_model="ResNet18"):
        super(ResNet, self).__init__()
        # model_resnet = resnet_dict[res_model](pretrained=True)
        model_resnet = resnet_dict[res_model](pretrained=False)
        # print (model_resnet)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        # print ('feature_layers:')
        # print (self.feature_layers)
        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)
        # print (self.hash_layer)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        return y

            
def tohex(code):
    code= str(code)
    code_hex = hex(int(code,2))[2:]
    
    if len(code_hex) < 4:
        code_hex = zero*(4-len(code_hex)) + code_hex
    return code_hex


def querylist(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = ResNet(16).to(device)
    model_dict= torch.load('./model/corel10k.pt')
    net.load_state_dict(model_dict)

    net.eval()

    
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                    ])

    img = Image.open(path).convert('RGB')
    img = transform(img).unsqueeze(0)
    qB = net(img).sign()[0].detach().numpy()
         
    qB_hex = tohex(qB)

    ganache_url = "http://127.0.0.1:8545"
    web3 = Web3(Web3.HTTPProvider(ganache_url, request_kwargs={'timeout': 100000000}))

    web3.eth.defaultAccount = web3.eth.accounts[3]

    #tx_hash = '0x96E19572073a5287Dd8CaD7E42a2A569561ee543'
    f = open(r"txhash.txt","r")
    tx_hash = f.read()
    # -----------------------------

    query = '0x' + str(qB_hex)
    topk = 10
    
    total = 0
    time,list1,list2 = searchTopk(web3, tx_hash, topk,query)
    total +=time
    print('-----------------------------------time :',total) 
    return list1,list2
