import torch

class Trainer():
    '''use for model training and testing '''
    def __init__(self,model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
    def train(self, train_loader, num_epochs=30, print_loss=True):
        print("Now training with epochos : {}".format(num_epochs))
        total_step = len(train_loader)
        model = self.model.to(self.device)
        for epoch in range(num_epochs):
            for i, (x, label) in enumerate(train_loader):
                x = x.to(self.device)
                label = label.to(self.device)
                # Forward pass
                outputs = model(x)
                loss = self.criterion(outputs, label)        
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()        
            if print_loss:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        return self
    
    def test(self,test_loader):
        print("Now testing...")
        self.model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for x, label in test_loader:
                x = x.to(self.device)
                label = label.to(self.device)
                outputs = self.model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

            print('Test Accuracy: {} %'.format(100 * correct / total))
        return self
    
    def save(self,path=None):
        if not path:
            path = '{}.model'.format(type(self.model).__name__)
        torch.save(self.model.state_dict(),path)
        