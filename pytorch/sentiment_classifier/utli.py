import torch
from itertools import dropwhile

def get_len(tensor,n=None):
    'find line length when packing rnn seqs (length without 0 padding)'
    if not n: n = tensor.shape[1]
    return torch.tensor([n-len(list(dropwhile(lambda x:x>0,i))) for i in tensor])

def sort_batch_collate(batch):
    data = torch.cat([item[0].unsqueeze(0) for item in batch])
    label = torch.cat([item[1].unsqueeze(0) for item in batch])
    _,indice = torch.sort(get_len(data),descending=True)    
    return (data[indice], label[indice])    

