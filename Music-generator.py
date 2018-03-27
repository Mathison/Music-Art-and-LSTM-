
# coding: utf-8

# In[2]:


import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# In[3]:


cuda=True


# In[4]:


##########read data separate by <start> and <end>
#########file represent the input file, batch represent batch size 
#########read_file is only used to read the data line by line in the tune

def read_file(file):
    with open(file,'r') as f:
        one_set=[]
        data=[]
        #line=readline(f)
        for l in f:
            '''
            if l.rstrip()=='<start>':  #####start index, we won't read <start> and <end> index
                continue
            '''
            one_set.append(l)
            if l.rstrip()=='<end>':  ####end of a set
                data.append(one_set)
                one_set=[]
                continue            
        return data
'''
seperate_data_hb is used to seperate the data into header part and body part 
'''
def seperate_data_hb(data):
    header=[]
    body=[]
    for d in data:
        head_set=[]
        body_set=[]
        for item in d:
            #########if it's <start>\r\n or the second character is : then it's header
            if item=='<start>\r\n' or item[1]==':':
                head_set.append(item)
            else:
                body_set.append(item)
        header.append(head_set)
        body.append(body_set)
    return header,body
'''
generate_data is used to get the character from original data file

seperate_data(data,batch) is used to get the character from batch to batch
data represent the data set we get from read_file
batch represent the batch size
'''
###########here is the function for a rough model, here we generate all the data together
def generate_data(data):
    data_set=[]
    ############put all character together
    for d in data:
        for item in d:
            char=list(item)
            data_set.extend(char)
    return data_set

def separate_data(data,batch):
    data_set=[]
    i=0
    while i+batch<len(data):
        data_set.append(data[i:i+batch])
        i+=batch
    '''
    if i<len(data):
        empty=['' for j in range(i+batch-len(data))]
        target=data[i:len(data)]
        target.extend(empty)
        data_set.append(target)
    '''
    data_set.append(data[i:len(data)])
    return data_set
'''
get_dict function is used to get the classes in the input file
classes represent the character, dict_data represent the index we gave for each character
matric_data represent the hot vector for each character
'''
def get_dict(data):
    data=list(set(data))
    dict_char={}
    dict_num={}
    matrix_data=[[0 for i in range(len(data))] for j in range(len(data))]
    num=0
    for d in data:
        dict_char[d]=num         #######use character as index
        dict_num[num]=d          #######use the number as index
        matrix_data[num][num]=1
        num+=1
    return dict_char,dict_num,matrix_data


# In[5]:


input_data=read_file('input.txt')


# In[6]:


char2idx,idx2char,class_matrix=get_dict(generate_data(input_data))


# In[7]:


##############seperate to training(80%) and validation(20%) set
t=int(len(input_data)*0.8)
train_data=generate_data(input_data[:t])
vali_data=generate_data(input_data[t:])


# In[8]:


#######here batch size is 30 but we may need to change it to increase the accuracy 
train_batch=separate_data(train_data,20)
vali_batch=separate_data(vali_data,20)


# In[9]:


import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

class Net(nn.Module):
    #####the layer should cntain the size for each layer including input and output
    #####the form of layer should like [1,....,size of class]
    def __init__(self, dropout=0, hidden_size=100, num_layers=1, Embedding_size=len(char2idx),type_model='LSTM'):
        super(Net, self).__init__()
        
        self.word_embeddings = nn.Embedding(len(char2idx), Embedding_size)

#         layers=[]
#         for l in range(len(layer)):
#             if l+2==len(layer):
#                 break
#             layers.append(nn.GRU(input_size=layer[l],
#                             hidden_size=layer[l+1],
#                             num_layers=1))
#         self.rnn=nn.Sequential(*layers)
        if type_model=="LSTM":
            self.rnn=nn.LSTM(input_size=len(char2idx),
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout)
        else:
            self.rnn=nn.GRU(input_size=len(char2idx),
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout)
        
        self.dense1 = nn.Linear(hidden_size,94)
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.type_model=type_model
        self.init_hidden()
        
    def forward(self, sequence):
        sequence=var(torch.LongTensor([char2idx[x] for x in sequence]))   ####get the index for character x
        embeds = self.word_embeddings(sequence)
        #x, self.hidden = self.rnn(embeds.view(len(sequence), 1, -1), self.hidden)
        x, new_hidden = self.rnn(embeds.view(len(sequence), 1, -1), self.hidden)
        x = x.view(-1, self.hidden_size)
        x = self.dense1(x)
        return x,new_hidden

    def init_hidden(self):
        if self.type_model=="LSTM":
            self.hidden = (Variable(torch.zeros(self.num_layers,1,self.hidden_size).cuda())
                        ,Variable(torch.zeros(self.num_layers,1,self.hidden_size).cuda()))
        else:
            self.hidden = Variable(torch.zeros(self.num_layers,1,self.hidden_size).cuda())

def var(x):
    x = Variable(x)
    if cuda:
        return x.cuda()
    else:
        return x


# In[10]:


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


def train(model, train_batch, vali_batch, optimizer, maxIteration=1000, cuda=True):
    
    criterion = nn.CrossEntropyLoss()
    train_result =[[],[]] # train_result[0]: loss; train_result[1]: accuracy
    vali_result =[[], []] # same as above
    for epoch in range(maxIteration):
        running_loss = 0
        running_acc = 0
        
        start = time.time()
        for minibatch in train_batch:
            

            
            """forward"""
            x, labels = minibatch[:-1], minibatch[1:]
            y, new_hidden = model(x)
            _, preds = torch.max(y.data, 1)
            labels = var(torch.LongTensor([char2idx[label] for label in labels]))
            
            """backprop"""
            loss = criterion(y, labels)
            loss.backward(retain_graph=False)
            optimizer.step()
            
            running_loss += loss.data[0]
            running_acc += torch.sum(preds == labels.data)
            
        end = time.time()
        
        """training loss and accuracy"""
        running_loss /= len(train_batch)
        running_acc /= float(len(train_data))
        train_result[0].append(running_loss)
        train_result[1].append(running_acc)
        
        """validation loss and accuracy"""
        vali_loss, vali_acc = evaluate(model, criterion, vali_batch)
        vali_result[0].append(vali_loss)
        vali_result[1].append(vali_acc)
        
        """print reuslts"""
        print('epoch[%s], time: %.3f, train loss: %.5f, train acc: %.5f, val loss: %.5f, val acc:%.5f' %
              (epoch, end-start, running_loss, running_acc, vali_loss, vali_acc))
        
        ############whenit keep increasing, break
        
        if epoch>6 and vali_result[0][-1] > vali_result[0][-5]:
            break
        
    x = range(len(train_result[0]))
    plt.figure(0)
    plt.ylabel('Loss', fontsize=20)
    plt.plot(x, train_result[0], label='train')
    plt.plot(x, vali_result[0], label='valid')
    print('final loss: ', train_result[0][-1], vali_result[0][-1])
    plt.legend()
    plt.grid()
    
    plt.figure(1)
    plt.ylabel('Accuracy', fontsize=20)
    plt.plot(x, train_result[1], label='train')
    plt.plot(x, vali_result[1], label='valid')
    print('final acc: ', train_result[1][-1], vali_result[1][-1])
    plt.legend()
    plt.grid()
    
    plt.show()

def evaluate(model, criterion, vali_batch):
    running_loss = 0
    running_acc = 0
    for minibatch in vali_batch:
        x, labels = minibatch[:-1], minibatch[1:]
        y,hidden = model(x)
        _, preds = torch.max(y.data, 1)
        labels = var(torch.LongTensor([char2idx[label] for label in labels]))
        loss = criterion(y, labels)
        running_loss += loss.data[0]
        running_acc += torch.sum(preds == labels.data)
    return running_loss/len(vali_batch), running_acc/float(len(vali_data))


# In[11]:


import random
from random import *
from datetime import datetime
seed(datetime.now())

from torch.autograd import Variable
def generate(starts, idx2char, predictLen, temperature, model):
    if len(starts) > 0:
        heat_map1=[]
        heat_map2=[]
        for count in range(predictLen):
            accWeight = []
            ## assume nextLetter is the best predicted vector of letter after alg.
            output,new_hidden = model(starts)
            output = F.softmax(torch.div(output, temperature))
            ########################get data from hidden layer to heatmap
            heat_map1.append(new_hidden[0].data[0][0])
            heat_map2.append(new_hidden[1].data[0][0])
            nextLetter = list((output[-1]).data)
            #print(nextLetter)
            accWeight = np.cumsum(nextLetter)
            #print(accWeight[0])
            #break
            prob = random()
            for i in range(len(idx2char)):           
                if (accWeight[i] >= prob):
                    starts += idx2char[i]
#                     res.append(idx2char[i])
                    break
#             starts += idx2char[i]
            l = len(starts)
            if (l >= 5):
                if  (starts[l - 1] == '>' and
                    starts[l - 2] == 'd' and
                    starts[l - 3] == 'n' and
                    starts[l - 4] == 'e' and
                    starts[l - 5] == '<'):
                    break
    return starts,heat_map1,heat_map2


# In[17]:


################initialize model
#######initial value self, dropout=0, hidden_size=100, num_layers=1, Embedding_size=len(char2idx),type_model='LSTM'
############here we test the plot when hidden_size=75 num_layers=1
model = Net(hidden_size=75,num_layers=1)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
if cuda:
    model.cuda()
train(model, train_batch, vali_batch, optimizer, maxIteration=50, cuda=True)


# In[18]:


################initialize model
#######initial value self, dropout=0, hidden_size=100, num_layers=1, Embedding_size=len(char2idx),type_model='LSTM'
############here we test the plot when hidden_size=75 num_layers=2
model = Net(hidden_size=75,num_layers=2)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
if cuda:
    model.cuda()
train(model, train_batch, vali_batch, optimizer, maxIteration=50, cuda=True)


# In[12]:


################initialize model
#######initial value self, dropout=0, hidden_size=100, num_layers=1, Embedding_size=len(char2idx),type_model='LSTM'
############here we test the plot when hidden_size=150 num_layers=1 dropout=0.1
model = Net(dropout=0.1,hidden_size=150,num_layers=1)
optimizer = optim.Adam(model.parameters(), lr=0.000005)
if cuda:
    model.cuda()
train(model, train_batch, vali_batch, optimizer, maxIteration=50, cuda=True)


# In[11]:


############here we test the plot when hidden_size=150 num_layers=1 dropout=0.1 optimizer=RMSprop
model = Net(dropout=0.1,hidden_size=150,num_layers=1)
optimizer_R=optim.RMSprop(model.parameters(), lr=0.000005)
if cuda:
    model.cuda()
train(model, train_batch, vali_batch, optimizer_R, maxIteration=50, cuda=True)


# In[ ]:


############here we test the plot when hidden_size=150 num_layers=1 dropout=0.1 optimizer=Adagrad
model = Net(dropout=0.1,hidden_size=150,num_layers=1)
if cuda:
    model.cuda()
optimizer_R=optim.Adagrad(model.parameters(), lr=0.000005)
train(model, train_batch, vali_batch, optimizer_R, maxIteration=50, cuda=True)


# In[12]:


############here we test the plot when hidden_size=150 num_layers=1 dropout=0.1 optimizer=Adam batchsize=20
train_batch=separate_data(train_data,20)
vali_batch=separate_data(vali_data,20)
model = Net(dropout=0.1,hidden_size=150,num_layers=1)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
if cuda:
    model.cuda()
train(model, train_batch, vali_batch, optimizer, maxIteration=50, cuda=True)


# In[13]:


############here we test the plot when hidden_size=150 num_layers=1 dropout=0.1 optimizer=Adam batchsize=100
train_batch=separate_data(train_data,100)
vali_batch=separate_data(vali_data,100)
model = Net(dropout=0.1,hidden_size=150,num_layers=1)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
if cuda:
    model.cuda()
train(model, train_batch, vali_batch, optimizer, maxIteration=50, cuda=True)


# In[52]:


############here we test the plot when hidden_size=150 num_layers=1 dropout=0.1 optimizer=Adagrad batchsize=100 learning rate=0.001
train_batch=separate_data(train_data,100)
vali_batch=separate_data(vali_data,100)
model = Net(dropout=0.1,hidden_size=150,num_layers=1)
if cuda:
    model.cuda()
optimizer = optim.Adagrad(model.parameters(), lr=0.001)
train(model, train_batch, vali_batch, optimizer, maxIteration=50, cuda=True)


# In[23]:


############here we test the plot when hidden_size=150 num_layers=1 dropout=0.1 optimizer=Adagrad batchsize=100 learning rate=0.001
############use 90% of data
train_batch=separate_data(train_data,100)
vali_batch=separate_data(vali_data,100)
model = Net(dropout=0.1,hidden_size=150,num_layers=1)
if cuda:
    model.cuda()
optimizer = optim.Adagrad(model.parameters(), lr=0.001)
train(model, train_batch, vali_batch, optimizer, maxIteration=50, cuda=True)


# In[15]:


############here we test the plot when hidden_size=150 num_layers=1 dropout=0.1 optimizer=Adam batchsize=100 learning rate=0.001
############use 90% of data
train_batch=separate_data(train_data,100)
vali_batch=separate_data(vali_data,100)
model = Net(dropout=0.1,hidden_size=150,num_layers=1)
if cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.00005)
train(model, train_batch, vali_batch, optimizer, maxIteration=50, cuda=True)


# In[71]:


m=model
music,heat_map1,heat_map2=generate('<start>',idx2char,1000,2, m)


# In[72]:


print(music)


# In[68]:


print(music)


# In[48]:


music_d=generate_data(music)
def get_heat(heat_map):
    heat=[[] for i in range(len(heat_map[0]))]
    count=0
    for h in heat:
        for d in heat_map:
            h.append(d[count])
        count+=1
    return heat


# In[49]:


h1=get_heat(heat_map1)
h2=get_heat(heat_map2)


# In[50]:


def heatmap(data, title, xlabel, ylabel):
    plt.figure()
    plt.figure(figsize=(50,70))
    plt.title(title,fontsize=20)
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel(ylabel,fontsize=20)
    c = plt.pcolor(data,  linewidths=4,cmap='RdBu', vmin=data.min(), vmax=data.max())
    return c
    #plt.colorbar(c)


# In[77]:


#h_map=(np.array(h1[1])+np.array(h2[1])).reshape(25,18)
h_map=np.array(h2[10]).reshape(25,18)
c=heatmap(h_map, 'heatmap for first nerum', 'x', 'y')


# In[78]:


def plot_heat(pc,h_map,music_d):
    char=0
    for y in range(h_map.shape[0]):
        for x in range(h_map.shape[1]):
            plt.text(x + 0.5, y + 0.5, music_d[char],
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=29
                     )
            char+=1
            
    plt.colorbar(pc)

    plt.show()


# In[79]:


plot_heat(c,h_map,music_d)


# In[200]:


plot_heat(c,h_map,music_d)


# In[201]:


print(music)

