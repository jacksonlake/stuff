
from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# Ignore warnings
import warnings

#self,sepal_length,sepal_width,petal_length,petal_width,flower_type

class FlowersDataBase:
    def __init__(self, data_file):
        self.data_file = data_file
        self.DB_raw = pd.read_csv(self.data_file,header=None)
        self.sepal_length = self.DB_raw[0]
        self.sepal_width = self.DB_raw[1]
        self.petal_length = self.DB_raw[2]  
        self.petal_width = self.DB_raw[3]
        self.flower_type = self.DB_raw[4]            

    def get_flower(self,ENTRY_NUMBER):
        #print(self.sepal_length[ENTRY_NUMBER], self.sepal_width[ENTRY_NUMBER], self.petal_length[ENTRY_NUMBER], self.petal_width[ENTRY_NUMBER], self.flower_type[ENTRY_NUMBER])
        return self.sepal_length[ENTRY_NUMBER], self.sepal_width[ENTRY_NUMBER], self.petal_length[ENTRY_NUMBER], self.petal_width[ENTRY_NUMBER], self.flower_type[ENTRY_NUMBER]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4 * 1, 32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32,3)

    def forward(self, x):
        x = F.relu(self.fc1(x))    
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.sigmoid(x)


def main():
       # interactive mode
    warnings.filterwarnings("ignore")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    training_example = torch.Tensor(1,4)                 #4x1 tensor  
    flowers = FlowersDataBase('iris.data') #create database
    net = Net()   #create neural network
    print(net)
    #print(net.forward(training_example))
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # create a stochastic gradient descent optimizer
    criterion = nn.MSELoss() #create a loss function
    
    net_out = net(training_example) #forward pass
    print('Original net output for training example #150', net_out  )

    target = torch.Tensor(1,3)
    
    for epoch in range(5000):                             #iterator over database
        running_loss = 0.0
        print('Epoch#:',epoch)

        for i in range(0,150):

            training_example[0,0], training_example[0,1], training_example[0,2], training_example[0,3], class_name = flowers.get_flower(i) 
            print('Example #:',i+1, training_example, class_name)
            optimizer.zero_grad()
            # forward + backward + optimize
            if(class_name=='Iris-virginica'):
                target[0,2] = 1
            if(class_name=='Iris-versicolor'):
                target[0,1] = 1
            if(class_name=='Iris-setosa'):
                target[0,0] = 1
            net_cur_output = net(training_example)
            loss = criterion(net_cur_output,target)
            print('Current net output = ',net_cur_output, 'Target output = ', target)
            print('Calculated Loss = ', loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print('running_loss = ', running_loss)

            target.fill_(0)
        print('New net output for training example #150',net(training_example))
        print('Training example was:',training_example)




    


if __name__ == '__main__':
    main()
