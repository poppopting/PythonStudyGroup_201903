import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CNN_sent import CNN_sent
from my_dataset import MyDataset
from trainer import Trainer
from itertools import islice


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
    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)

    lines,labels = label_line_io(test_path)
    test_dataset = MyDataset(lines,labels,pad_zero=True)
    test_loader = DataLoader(test_dataset,batch_size=64,shuffle=True)
    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = CNN_sent(num_words=train_dataset.wordcounts,
                     max_line_len=train_dataset.max_line_length,
                     emb_dim=100,conv_heights=[3,5,7],out_channels=[20,20,20])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 30

    trainer = Trainer(model,criterion,optimizer,device)
    trainer.train(train_loader,num_epochs=30)
    trainer.test(test_loader)
    trainer.save()

if __name__ == '__main__':
    main()

