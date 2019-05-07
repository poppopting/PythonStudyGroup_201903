import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from RNN_sent import  RNN_sent
from my_dataset import MyDataset
from trainer import Trainer
from itertools import islice
from utli import get_len, sort_batch_collate


train_path = 'data/sentiment_XS_30k.txt'
test_path = 'data/sentiment_XS_test.txt'


def label_line_io(path,encoding):
    'input data'
    labels = []
    lines = []
    with open(path,'rt',encoding=encoding) as f:
        for l in islice(f,1,None):
            lab,line = l.strip("\n").split(",")
            if len(line.split(" "))>10:
                continue
            if len(line.split(" "))<3:
                continue                            
            lines.append(line)
            labels.append(lab)
    return lines,labels    

def main():
    lines,labels = label_line_io(train_path)
    train_dataset = MyDataset(lines,labels,pad_zero=True)
    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True,collate_fn=sort_batch_collate)

    lines,labels = label_line_io(test_path)
    test_dataset = MyDataset(lines,labels,pad_zero=True)
    test_loader = DataLoader(test_dataset,batch_size=64,shuffle=True,collate_fn=sort_batch_collate)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = RNN_sent(num_words=train_dataset.wordcounts,
                     max_line_len=train_dataset.max_line_length,
                     RNN_cell='gru',
                     device=device,
                     emb_dim=100,
                     hidden_dim=40,
                     bidirectional=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    Trainer(model,criterion,optimizer,device).train(train_loader,num_epochs=30).test(test_loader).save()

if __name__ == '__main__':
    main()

