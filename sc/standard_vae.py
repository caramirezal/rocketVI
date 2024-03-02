## Testing Architectures of VAE. Based in: 
## https://avandekleut.github.io/vae/

## In order to run this script activate the docker container using
#docker run -it -p 8080:8080 --gpus 1 --rm -v  /raid/:$HOME pytorch/std_vae

## Dependencies
import os
import sys
import scanpy as sc
import torch 
import torch.nn as nn
import matplotlib
from matplotlib import pyplot as plt
import torch.optim as optim
import umap
import numpy as np


## Path in the curry cluster
## path2project = "/home/bq_cramirez/cramirez/rocketVI"

## Path in the dgx server
path2project = "/workspace/rocketVI/"

os.chdir(path2project)
os.getcwd()

## Setting up the cuda gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## Dependencies
## path in the dgx curry station
scRNA_datapath = '/workspace/rocketVI/data/pbmc/' 
adata = sc.read(scRNA_datapath + 'PBMC_train_preprocessed.h5ad')


## Exploratory analysis of the data
## Plotting UMAP
path2figures = path2project + 'figures'
if not os.path.exists(path2figures):
    os.makedirs(path2figures)

path2figures = path2figures + '/standard_vae'
if not os.path.exists(path2figures):
    os.makedirs(path2figures)



with plt.rc_context():  # Use this to set figure params like size and dpi
    sc.pl.umap(adata, color='cell_type', show=False)
    plt.savefig(path2figures + "/umap_PBMC_train_preprocessed.pdf", bbox_inches="tight")
    plt.clf()




gex_tensor = torch.tensor(adata.X.todense())
gex_tensor



## Autoencoder
class enc(nn.Module):
    def __init__(self):
        super(enc, self).__init__()
        self.layer1 = nn.Linear(adata.n_vars, 4000)
        self.drop = nn.Dropout(p=0.1)
        self.layer2 = nn.Linear(4000, 500)
    
    def forward(self, mtx):
        mtx_out1 = self.layer1(mtx)
        mtx_out1 = self.drop(mtx_out1)
        mtx_out2 = self.layer2(mtx_out1)
        mtx_out2 = self.drop(mtx_out2)
        return mtx_out1, mtx_out2       ## a way to retrieve layer outputs
    


gex_tensor.shape[1]


## Instantiate model
model1 = enc()

## Run the model
res = model1(gex_tensor)


## Defining the optimizer
optimizer = optim.SGD(model1.parameters(), lr=0.001, momentum=0.9)


## Inspecting the network state
model1.state_dict()
for parameter in model1.state_dict():
    print(parameter, ': Size=', model1.state_dict()[parameter].size())


"""
layer1.weight : Size= torch.Size([4000, 6998])
layer1.bias : Size= torch.Size([4000])
layer2.weight : Size= torch.Size([500, 4000])
layer2.bias : Size= torch.Size([500])
layer3.weight : Size= torch.Size([30, 500])
layer3.bias : Size= torch.Size([30])'
"""
## Getting weights (for example) from model
model1.state_dict()['layer1.weight']          
## There is a similar method for the optimizer optimizer.state_dict()


## Visualizing layers with UMAP
## Function to project 2d matrices and plotting
reducer = umap.UMAP()
embedding = reducer.fit_transform(res[0].detach().numpy())
## res[0].detach().numpy() contains the output of the first layer




## Definition of the decoder
class dec(nn.Module):
    def __init__(self):
        super(dec, self).__init__()
        self.linear1 = nn.Linear(500, 4000)
        self.linear2 = nn.Linear(4000, adata.n_vars)
    
    def forward(self, mtx):
        mtx_out1 = self.linear1(mtx)
        mtx_out2 = self.linear2(mtx_out1)
        return mtx_out1, mtx_out2
    


## Testing the decoder



## Testing the decoder
dec1 = dec()
res_dec = dec1(res[1])
res_dec.shape


## definition of the autoencoder
class encoder_encoder(nn.Module):
    def __init__(self):
        super(encoder_encoder, self).__init__()
        self.enc = enc()
        self.dec = dec()
    
    def forward(self, mtx):
        z = self.enc(mtx)
        output = self.dec(z[1])
        return output




## testing the encoder_decoder
auto_enc = encoder_encoder()
res_ae = auto_enc(gex_tensor)
auto_enc

res_ae[1].shape
