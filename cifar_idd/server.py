from flask import Flask, request
import json
import requests
import ast
import threading

import os
import random
from tqdm import tqdm
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset   
torch.backends.cudnn.benchmark=True
import csv
#################################
##### Usefull variables #####
#################################

global total_files_gatherd
global fl_round 

total_files_gatherd =0
fl_round =0
threads = []


def scv_write_results(fr,acc,loss):
    with open("results.csv", mode='a') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow([str(fr),str(acc),str(loss)])
  
#################################
##### Neural Network model #####
#################################

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def dispatch_model(url,files):
  req = requests.post(url=url,files=files)
  return 1


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        output = F.log_softmax(out, dim=1)
        return output

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)




def client_update(client_model, optimizer, train_loader, epoch=5):
    """
    This function updates/trains client model on client data
    """
    model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data, target
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()



def server_aggregate(global_model, client_models):
    """
    This function has aggregation method 'mean'
    """
    ### This will take simple mean of the weights of models ###
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
        global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())



def test(global_model, test_loader):
    """This function test the global model on test data and returns test loss and test accuracy """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            output = global_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc

#############################################################
##### Hyperparameters  #####
#############################################################
num_clients = 4
num_selected = 4
num_rounds = 4
epochs = 5
batch_size = 32
global_model =  VGG('VGG19')



#############################################################
##### Creating desired data distribution among clients  #####
#############################################################
def create_data():

# Image augmentation 
  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Loading CIFAR10 using torchvision.datasets
  traindata = datasets.CIFAR10('./data', train=True, download=True,
                       transform= transform_train)

# Dividing the training data into num_clients, with each client having equal number of images
  traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / num_clients) for _ in range(num_clients)])


# Creating a pytorch loader for a Deep Learning model
  train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]


# Normalizing the test images
  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Loading the test iamges and thus converting them into a test_loader
  test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        ), batch_size=batch_size, shuffle=True)


  return (train_loader,test_loader)


############## client models ##############
client_models = [ VGG('VGG19') for _ in range(num_selected)]
for model in client_models:
    model.load_state_dict(global_model.state_dict()) ### initial synchronizing with global model 

############### optimizers ################
opt = [optim.SGD(model.parameters(), lr=0.1) for model in client_models]

losses_train = []
losses_test = []
acc_train = []
acc_test = []


#############################################################
##### FLASK interface  #####
#############################################################

app = Flask(__name__)

@app.route('/')
def hello():
  return "Server running !"

@app.route('/init')
def init():
  train_loader,test_loader = create_data()
  return "OK !"


@app.route('/start', methods=['GET', 'POST'])
def start():
  global threads
  train_loader,test_loader = create_data()

  
  torch.save(model.state_dict(), "global_model")
  f = open("global_model", 'rb')
  port = 8000
  for users in range(0,num_clients):
    f = open("global_model", 'rb')
    port = port + 1
      #Global model dispatch to the UEs
    data = {'fname':'global_model', 'id':str(users), 'federated round':str(1)}
    files = {
		    'json': ('json_data',json.dumps(data),'application/json'),
		    'model': ('global_model', f, 'application/octet-stream')}

    url = str("http://localhost:"+str(port)+"/send_model")
    #req = requests.post(url=url,files=files)

    thread = threading.Thread(target=dispatch_model, args=(url,files,)) 
    threads.append(thread)
    print (url)

  #f.close()
  for i in range (0,num_clients):
    print ("dispatching models")
    threads[i].start()
  threads = []
 
  return "Model sent !"

@app.route('/aggregate',methods=['POST'])
def aggregate():
  global total_files_gatherd
  global fl_round 
  global threads

  train_loader,test_loader = create_data()
  client_models_list =[]
  uploaded_file = request.files['model']
  fname = request.files['json'].read()
  fname = ast.literal_eval(fname.decode("utf-8"))
  user_id = int(fname['user_id'])
  loss = float(fname['loss'])
  model_name = str("aggregate_"+str(user_id))
  uploaded_file.save(model_name)

  total_files_gatherd = total_files_gatherd +1
  print ("files gathered "+ str(total_files_gatherd))
  #print("loss:"+ str(loss))
  if (total_files_gatherd == num_clients):

     for i in range (0,num_clients):
       #gather all client models in one list
       client_models_list.append(model.load_state_dict(torch.load(str("aggregate_"+str(i)))))

     print("aggregating models")
     
     if (fl_round != num_rounds):
       #aggregate models here
       server_aggregate(global_model, client_models)
       test_loss, acc = test(global_model, test_loader)
       print("Round: "+ str(fl_round))
       print("Acc: "+ str(acc))
       print("Loss: "+ str(test_loss))
       print("--------------------------")
       scv_write_results(fl_round,acc,test_loss)
       
       
       #dispatch models to users
       torch.save(global_model.state_dict(), "global_model")
       f = open("global_model", 'rb')
       port = 8000
       for users in range(0,num_clients):
        f = open("global_model", 'rb')
        port = port+1
       #Global model dispatch to the UEs
        data = {'fname':'global_model', 'id':str(users), 'federated round':str(fl_round)}
        files = {
		    'json': ('json_data',json.dumps(data),'application/json'),
		    'model': ('global_model', f, 'application/octet-stream')}

        url = str("http://localhost:"+str(port)+"/send_model")
        #req = requests.post(url=url,files=files)
            
        thread = threading.Thread(target=dispatch_model, args=(url,files,)) 
        threads.append(thread)
        #print (url)
        total_files_gatherd =0 
        print ("user model veryfied")

       #f.close()
       for i in range (0,num_clients):
         print ("dispatching models")
         threads[i].start()

       threads = []
       fl_round = fl_round +1

     else:
       print ("federated learning completed!")

  return "Model aggregated and sent back to users !"


@app.route('/testing',methods=['GET'])
def testing():
  global var
  var =var +1
  print (var)
  return "Model sent !"

if __name__ == '__main__':
  app.run(host='localhost', port=8000, debug=False, use_reloader=True)


