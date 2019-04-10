import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

class CNN_sent(nn.Module):   
    '''
    Conduct CNN sentence sentiment classification
    Reference : Convolutional Neural Networks for Sentence Classification

    Param
    -----
    num_words: num of words to emb
    max_line_len: the maximum length of lines. 
                  to deal with differnet line length, you may padding to same len, or use batch size=1 
    emb_dim: number of dims for word embedding
    conv_heights: list of int for heights of conv layers. If you only want single one height, pass a list anyway
    out_channels: list of int. number of feature maps for different conv layers. length should be equal to conv_heights
    pre_train_emb: pre-trained embedding feature, default None

    '''
    def __init__(self,
                 num_words:int ,
                 max_line_len:int,
                 emb_dim:int, 
                 conv_heights:list, 
                 out_channels:list,
                 pre_train_emb=None): 
        assert len(conv_heights)==len(out_channels)       
        super().__init__()
        conv_out_channel = reduce(lambda x,y:x+y,out_channels)
        
        if pre_train_emb:
            self.embedding = nn.Embedding.from_pretrained(pre_train_emb)
        else:
            self.embedding = nn.Embedding(num_words,emb_dim)
        self.conv_list = nn.ModuleList([nn.Conv2d(1,oc,(height,emb_dim)) 
                                            for oc,height in zip(out_channels,conv_heights) ])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.batch_norm = nn.BatchNorm1d(conv_out_channel)
        self.fc = nn.Linear(conv_out_channel,20)
        self.drop = nn.Dropout()
        self.output_layer = nn.Linear(20,2)                        
        
    def forward(self,batch_line):        
        x = self.embedding(batch_line).unsqueeze(1)
        tensor_list = []
        for conv in self.conv_list:
            cur = conv(x).squeeze(3)
            cur = self.pool(cur).squeeze(2)
            tensor_list.append(cur)
        x = torch.cat(tuple(tensor_list),dim=1)
        x = self.batch_norm(x)
        x = self.fc(x)  
        x = F.relu(x)
        x = self.drop(x)
        x = self.output_layer(x)
        x = F.softmax(x,dim=1)
        return x  

