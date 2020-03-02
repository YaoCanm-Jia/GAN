# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:41:46 2019

@author: Yaocm
"""

#导入所需要的库
import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets

transform=transforms.Compose([
    transforms.Resize(64), #缩放图片，最短边的长为64像素,
    transforms.CenterCrop(64), #从中间切出64*64的图片
    transforms.ToTensor(), 
    transforms.Normalize([0.5,0.5,.5],[0.5,0.5,0.5]) 
])


#定义自己的数据集合
class Flower(data.Dataset):
    
    def __init__(self,root,transform):
        #所有图片的绝对路径
        imgs=os.listdir(root)
        self.imgs=[os.path.join(root,k) for k in imgs]
        self.transforms=transform

    def __getitem__(self, index):
        img_path=self.imgs[index]
        #定义标签---->flower
        label='flower'
        pil_img=Image.open(img_path)
        if self.transforms:
            data=self.transforms(pil_img)
        return data,label

    def __len__(self):
        return len(self.imgs)

#dataSet=Flower('E:/总目录/大三下实习/oxford102/',transform=transform)
#
#print(dataSet[0])
#
#
#
#dataset1 = datasets.MNIST(root='E:/总目录/大三下实习/data', download=True,train = True,
#                          transform=transforms.Compose([
#                       transforms.Resize(64),
#                       transforms.ToTensor(),
#                       transforms.Normalize([0.5], [0.5]),
#                   ]))
#
#print(dataset1[0])

##print(dataSet[1])
##print(dataSet[2])
##print(dataSet[3])
##print(dataSet[4])
#
#dataloader = torch.utils.data.DataLoader(dataSet, batch_size=64,
#                                         shuffle=True)
#
#dataiter = iter(dataloader)
#
#imgs, labels = next(dataiter)
#print(imgs.size()) 
##
#for i, (data, _) in enumerate(dataloader):
#    print(i,data)