import torch
from torch.utils.data import DataLoader,Dataset

class MyDataset(Dataset):
    '''process data and set as a pytorch.utils.data.Dataset.
       
       Input:
       ----
       lines    : list of line, with line in string format.
       labels   : list of labels, label could be int or string.
       pad_zero : whether to padding zero to lines, making all lines equal length. Default true.    
    '''
    def __init__(self,lines:list,labels:list,pad_zero=True):
        assert len(lines) == len(labels), "number oflabels and lines not equal"
        self.word2index = dict()
        self.index2word = dict()
        self.indexlines = list()
        self.wordcounts = 0
        self.max_line_length = 0
        self.labels = list() 
        self.label_mapping = dict()        
        self._word_to_index(lines,pad_zero)
        self._label_to_index(labels)
        
    def _word_to_index(self,lines,pad_zero):
        'get word and index mapping'
        if pad_zero:
            self.word2index['emptyWord'] = 0
            self.index2word[0] = 'emptyWord'                    
        cur_index = len(self.word2index)
        for l in lines:
            words = l.strip("\n").split(" ")
            for word in words:
                if word not in self.word2index:
                    self.word2index[word] = cur_index
                    self.index2word[cur_index] = word
                    cur_index += 1
            self.indexlines.append([self.word2index[word] for word in words])
        self.wordcounts = cur_index
        self.max_line_length = len(max(train_dataset.indexlines,key=lambda x:len(x)))
        if pad_zero:
            self.indexlines = [line+[0]*(self.max_line_length-len(line)) 
                                        for line in self.indexlines]
        
    def _label_to_index(self,labels):
        'transform labels to int labels'
        label2index = {lab:no for no,lab in enumerate(set(labels))}
        self.labels = [label2index[label] for label in labels]
        self.label_mapping = label2index
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,index):
        if isinstance(index,slice):
            raise TypeError('Doesnt support indexing')
        return (torch.tensor(self.indexlines[index]),
                torch.tensor(self.labels[index]))