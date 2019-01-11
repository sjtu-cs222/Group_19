
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# from module import Net
from VGG16_mnist import *
from VGG16_mnist import DataSet
import pickle as pkl


class activationHooker:
    def __init__(self,model):
        # self.hookdist = {}
        self.activations = {}
        self.groundTruth = []
        for i in range(10):
            self.activations[i] = {}
        self.model = model
        self.model.apply(lambda m: self.reg_hook(m,lambda m,inp,outp: self.hook(m,inp,outp,self.activations)))

    
    def reg_hook(self,m,hook):
        if(isinstance(m, torch.nn.modules.conv.Conv2d)):
            print(m)
            m.register_forward_hook(hook)

    def hook(self,m,inp,outp,stor):
        assert(len(self.groundTruth)==outp.shape[0])
        #print(m,outp.shape,len(stor[0].keys()))
        if(True):
            for n, atv in zip(self.groundTruth,outp):
                n = int(n)
                activation = abs(atv).max(1)[0].max(1)[0]
                #print(activation.shape)
                assert(len(activation.shape)==1)
                if(m in stor[n].keys()):
                    act = stor[n][m]
                    act += activation.cpu().numpy()# get the average of maximum activation
                    #act.append(activation.cpu().numpy() ) # get some samples about the activation
                else:
                    act = activation.cpu().numpy()
                    #act = [activation.cpu().numpy()]
                stor[n][m] = act

    def dataloaders(self,data):
        #return dataloaders of different number (number,dataloader)
        # datas = {}
        # for dt, tg in data:
        #     datas{tg}
        pass
    
    def analysisACT(self,loader,device):
        self.model.eval()
        count = 0
        with torch.no_grad(): 
            for i, (batch, label) in enumerate(loader):
                batch = batch.to(device)
                label = label.to(device)
                # print(target)
                # self.hookdist = self.activations[int(target.numpy())] # change the pointer to make ti
                self.groundTruth = label
                
                output = self.model(batch)
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                count+=1
                    # print("a")
                if(i%200==199):
                   print(i,label)
                   #break       
            tempdict = {}
        # for lay in self.hookdist.keys():
            for i in range(10):
                tempdict[i] = {}
                for name, act in self.model.named_modules():
                    if(isinstance(act, torch.nn.modules.conv.Conv2d)):
                        print(name, act)
                        tempdict[i][name] = self.activations[i][act]/count
                        #tempdict[i][name] = self.activations[i][act]
                # tempdict[lay] = self.hookdist[lay]/count
            #self.hookdist = {}
            print(tempdict)
        
            return tempdict

if __name__ == "__main__":
    test_path = "./mnist_data"
    use_cuda = torch.cuda.is_available()
    pin_memory = True
    # torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    dataset=DataSet(torch_v=0.4)
    test_loader = dataset.test_loader(test_path,pin_memory=pin_memory)

    model = torch.load("vgg16_model_mnist").to(device)
    hooker = activationHooker(model)
    tempdist = hooker.analysisACT(test_loader,device)
    with open("tempdist.pkl","wb") as f:
        pkl.dump(tempdist,f)

    # only these things are pickable
    # functions defined at the top level of a module (using def, not >lambda)
    # built-in functions defined at the top level of a module
    # classes that are defined at the top level of a module

