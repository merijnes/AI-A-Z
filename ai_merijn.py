# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:32:44 2019

@author: merijn

From: Course Aritifical Intelligence A-Z
"""

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Create the Architecture of the Neural Network

class Network(nn.Module):
    
    def __init__(self,input_size,nb_action): # initialize the Neural Network
        super(Network, self).__init__()
        self.input_size = input_size        # input neuron, size 5
        self.nb_action = nb_action          # output neuron, size 3
        self.fc1 = nn.Linear(input_size,15) # full connection input layer to hidden layer of 30 neurons
        self.fc2 = nn.Linear(15,30)         # full connection of hidden layer of 30 neurons to hidden layer
        self.fc3 = nn.Linear(30,30)         # full connection of hidden layer of 30 neurons to hidden layer
        self.fc4 = nn.Linear(30,15)         # full connection of hidden layer of 30 neurons to hidden layer
        self.fc5 = nn.Linear(15,nb_action)  # full connection of hidden layer of 30 neurons to output layer
        
    def forward(self, state):               # forward function
        x1 = F.relu(self.fc1(state))        # Rectifier function, activates the hidden neurons
        x2 = F.relu(self.fc2(x1))           # Rectifier function, activates the hidden neurons
        x3 = F.relu(self.fc3(x2))           # Rectifier function, activates the hidden neurons
        x4 = F.relu(self.fc4(x3))           # Rectifier function, activates the hidden neurons
        q_values = self.fc5(x4)             # get the output neurons
        return q_values                     # return q_values
    
# Implementing Experience Replay (longterm memory)
        
class ReplayMemory(object):
    
    def __init__(self, capacity):           # initialize replay object, capacity of 100
        self.capacity = capacity
        self.memory = []                    # initialize memory List with []
        
    def push(self, event):                  # push new event into memory
        self.memory.append(event)           # add  last state, new state, last action, last reward
        if len(self.memory) > self.capacity:     # check if memory > capacity to keep memory within capacity length
            del self.memory[0]              # remove the first element
   
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory,batch_size))  # take random samples of fixed batch size into memory, zip for specific structure
        return map(lambda x: Variable(torch.cat(x,0)), samples)  
    
# Deep Q Learning Algorithm
        
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []                         # initialize reward window list
        self.model = Network(input_size, nb_action)     # Creates a neural network with class Network
        self.memory = ReplayMemory(100000)              # Create memory with a certain capacity 
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.0001)  # connect optimizer Adam to neural network with learning rate
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*25) # temperature parameter = 7
        action = probs.multinomial()
        return action.data[0,0]
        
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        self.optimizer.step()
        
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
           
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...") 