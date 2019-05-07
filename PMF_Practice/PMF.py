

# edited by ting 
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np


# load file
data=defaultdict(dict)
with open('u.data','r') as movie:
    for record in movie:
        user_id,item_id,rating,timestamp=record.split("\t")
        data[int(user_id)][int(item_id)]=int(rating)



class PMF(nn.Module):
    def __init__(self,num_user:int,num_item:int,hidden_factors:int):
        super().__init__()
        self.num_user=num_user
        self.num_item=num_item
        self.hidden_factors=hidden_factors
        self.emb_user=nn.Embedding(num_user,hidden_factors,sparse=True)
        self.emb_item=nn.Embedding(num_item,hidden_factors,sparse=True)
        
    def forward(self,movie_matrix):
        self.user_u=self.emb_user(torch.LongTensor(torch.randint(np.long(1),np.long(5),(self.num_user,1))))
        self.item_v=self.emb_item(torch.LongTensor(torch.randint(np.long(1),np.long(5),(self.num_item,1))))
        
        predict_ratings=[float(torch.mm(self.user_u[user_-1].view(1,-1),(self.item_v[item_-1]).view(-1,1))) for user_ in movie_matrix.keys() for item_ in movie_matrix[user_].keys()]
        return F.relu(torch.Tensor(predict_ratings)).requires_grad_(True)


# take out ratings

ratings=[np.long(item_) for user_ in data.keys() for item_ in data[user_].values()]


# begin running
pmf=PMF(943,1682,20)
loss=nn.MSELoss()
sgd=torch.optim.SGD(pmf.parameters(),lr=1e-6,weight_decay=1e-3)
for i in range(10):
    predict=pmf(data)
    
    losses=loss(predict,torch.FloatTensor(ratings))#torch.Tensor(ratings)
    
    sgd.zero_grad()
    print(losses.data)
    losses.backward()
    sgd.step()
    print('-------')
    
