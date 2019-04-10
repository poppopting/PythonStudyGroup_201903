import torch
import torch.nn as nn
import torch.nn.functional as F
from utli import get_len
from functools import reduce


class RNN_sent(nn.Module):   
    '''
    Conduct CNN sentence sentiment classification
    Reference : Convolutional Neural Networks for Sentence Classification

    Param
    -----
    num_words: num of words to emb
    max_line_len: the maximum length of lines. 
                  to deal with differnet line length, you may padding to same len, or use batch size=1 
    emb_dim: number of dims for word embedding
    hidden_dim: dim of RNN hidden 
    device: whcih torch.device to use
    RNN_cell: should be one of 'lstm','gru','rnn', default 'lstm'
    pre_train_emb: pre-trained embedding feature, default None
    rnn_args : other args will throw into rnn layers

    '''
    def __init__(self,
                 num_words:int ,
                 max_line_len:int,
                 emb_dim:int,                 
                 hidden_dim:int,
                 device=None,
                 RNN_cell='lstm',                 
                 pre_train_emb=None,                 
                 **rnn_args): 
        
        super().__init__()        
        if pre_train_emb:
            self.embedding = nn.Embedding.from_pretrained(pre_train_emb)
        else:
            self.embedding = nn.Embedding(num_words,emb_dim,padding_idx=0)
        
        if not device:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')            
        self.device = device
            
        RNN = {'lstm': nn.LSTM(emb_dim,hidden_dim,batch_first=True,**rnn_args),
               'gru': nn.GRU(emb_dim,hidden_dim,batch_first=True,**rnn_args),
               'rnn': nn.RNN(emb_dim,hidden_dim,batch_first=True,**rnn_args)}
        try:
            self.rnn = RNN[RNN_cell]
            self.rnn_type = RNN_cell
        except KeyError:
            raise KeyError('RNN_cell should be "lstm" or "gru"! ')

        self.num_direction = rnn_args.get('bidirectional',0)+1 #true:2,false:1,None:1                            
        self.num_layer = rnn_args.get('num_layers',1)
        self.hidden_dim = hidden_dim
        self.drop = nn.Dropout()
        self.output_layer = nn.Linear(self.num_direction*hidden_dim,2)                        
        
    def forward(self,batch_line):  
        length = get_len(batch_line)
        batch_size = batch_line.shape[0]
        x = self.embedding(batch_line)#.unsqueeze(1)
        
        pack = nn.utils.rnn.pack_padded_sequence(x,length,batch_first=True)        
        h0 = torch.zeros((self.num_layer*self.num_direction,batch_size,self.hidden_dim)).to(self.device)

        if self.rnn_type == 'lstm':
            c0 = torch.zeros((self.num_layer*self.num_direction,batch_size,self.hidden_dim)).to(self.device)
            packed,(hn,cn) = self.rnn(pack,(h0,c0))
        else:
            packed,hn = self.rnn(pack,h0)
        x, x_lengths = nn.utils.rnn.pad_packed_sequence(packed)
        x = x[x_lengths-1,torch.arange(batch_size),:] # last state        
        x = F.relu(x)
        x = self.drop(x)
        x = self.output_layer(x)
        x = F.softmax(x,dim=1)
        return x  