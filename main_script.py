import numpy as np
import sklearn.metrics as metrics
import json
import os
from model_module import ModelBuilder
from ResNetTester_cls import ResNetTester,make
from Datasets import (cifer10_datasets,cifer100_datasets,mnist_dataset)


# 結果の保存名
name = "DoubleResNet_Test"

epochs = 1
split = 1.0
batch_size = 64
dataset = cifer10_datasets(is_zero_center=True)

test_option = {
    #--------------
    #Residual Block
    #--------------
    #"block": "double_basic",
    "block": "basic_block",
    #"block": "double_bottleneck"
    #"block": "bottleneck",

    "relu_option":False,

    #TrueにするとDouble ResNet
    "double_input":False,

    #TrueにするとWide ResNet
    "wide":False,

    #フィルター数(wide resnet用)
    #k=1では64,k=2では128
    "filters": 64,
    #"filters":128,

    #--------------
    #結合手法
    #--------------
    "concatenate": "half_concanate",
    #"concatenate": "hull_concanate",
    #"concatenate": "sum",
    #"concatenate": "none"

    "reseption" :[2,2,2,2],
    "dropout": 0
}

model, tester = make(test_option,dataset,batch_size,epochs,split)
tester.run(model,name)