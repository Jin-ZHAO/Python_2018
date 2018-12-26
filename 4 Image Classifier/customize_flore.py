
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
import json
from matplotlib.ticker import FormatStrFormatter
from collections import OrderedDict
import argparse


nThreads = 4
batch_size = 8
use_gpu = torch.cuda.is_available()


def load_data(where  = "./flowers" ):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_trans = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    validation_trans = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])
    test_trans = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_trans)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_trans)
    test_data = datasets.ImageFolder(test_dir ,transform = test_trans)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valiloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)
    
    return trainlodader, valiloader, testloader


# In[ ]:


def preparation(structure='densenet121',dropout=0.5, hidden_layer1 = 120,lr = 0.001):
    model = models.vgg19(pretrained=True)
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 102)),
                          #('relu2', nn.ReLU()),
                          #('fc3', nn.Linear(1048, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          #('dropout',nn.Dropout(p=0.5))
                          ]))
    for param in model.parameters():
        param.requires_grad = False   
    model.classifier = classifier
    model
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    return model, criterion,optimizer


# In[ ]:


def train_network(model, criterion, optimizer, epochs = 20, print_every=20, power='gpu'):
    steps = 0
    loss_show=[]

    print("--------------Training is starting------------- ")
    
    model.to('cuda')
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs,labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                vlost = 0
                accuracy=0
                for ii, (inputs2,labels2) in enumerate(valiloader):
                    optimizer.zero_grad()
                    inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                    model.to('cuda:0')
                    with torch.no_grad():    
                        outputs = model.forward(inputs2)
                        vlost = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
                vlost = vlost / len(valiloader)
                accuracy = accuracy /len(valiloader)
            
                    
            
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Lost {:.4f}".format(vlost),
                       "Accuracy: {:.4f}".format(accuracy))
            
            
                running_loss = 0
                
    print("-------------- Finished training -----------------------")


# In[ ]:


def save_checkpoint(path='checkpoint.pth', structure ='densenet121', hidden_layer1=120,dropout=0.5,lr=0.001,epochs=20):
    model.cpu()
    checkpoint = {'arch': 'vgg13',
              'class_to_idx':train_data.class_to_idx,
              'epochs': 20,
              'input_size' :25088,
              'output_size': 102,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'model_classfier': model.classifier.state_dict()}
    print(train_data.class_to_idx)
    torch.save(checkpoint, 'checkpoint.pth')


# In[ ]:


def load_checkpoint(path='checkpoint.pth'):
    checkpoint = torch.load(filepath,map_location='cpu')
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    
    
    #from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 102)),
                          #('relu2', nn.ReLU()),
                          #('fc3', nn.Linear(1048, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          #('dropout',nn.Dropout(p=0.5))
                          ]))
    
    model.classifier = classifier  
    
    if args.gpu:
        if use_gpu:
            model = model.cuda()
            print ("Using GPU: "+ str(use_gpu))
        else:
            print("Using CPU since GPU is not available")

    model.load_state_dict(checkpoint['state_dict'])
    return model


# In[ ]:


def process_image(image_path):
    the_img = Image.open(image)

    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(the_img)
    
    return tensor_image
                  


# In[ ]:


def predict(image_path, model, topk=5):   
    model.to('cuda:0')
    img_torch = process_image(img)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        
    probability = torch.exp(output.data)
    
    return probability.topk(topk)

