class CNN_sent(nn.Module):   
    def __init__(self,
                 num_words:int ,
                 max_line_len:int,
                 emb_dim:int, 
                 conv_heights:list, 
                 out_channels:list,
                 pre_train_emb=None, 
                 use_cuda=False):        
        super().__init__()
        conv_out_channel = reduce(lambda x,y:x+y,out_channels)
        self.embedding = nn.Embedding(num_words,emb_dim)
        self.conv_list = nn.ModuleList([nn.Conv2d(1,oc,(height,emb_dim)) 
                                            for oc,height in zip(out_channels,conv_heights) ])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(conv_out_channel,10)
        self.output_layer = nn.Linear(10,2)                        
        
    def forward(self,batch_line):
        x = self.embedding(batch_line).unsqueeze(1)
        tensor_list = []
        for conv in self.conv_list:
            cur = conv(x).squeeze(3)
            cur = self.pool(cur).squeeze(2)
            tensor_list.append(cur)
        x = torch.cat(tuple(tensor_list),dim=1)
        x = self.fc(x)  
        x = F.relu(x)
        x = self.output_layer(x)
        x = F.softmax(x,dim=1)
        return x                                    
