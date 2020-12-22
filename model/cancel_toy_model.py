import numpy as np
import torch

from utils import save_checkpoint


class CancelModel(torch.nn.Module):
    """ A toy model (1 hidden layer preceptron) for predicting 'is_cancelled' (0/1)
    """
    def __init__(self, save_model_name, input_size, hidden_size, epoch, optimizer, loss, lr, seed, use_cuda=False, **other_params):
        torch.manual_seed(seed)
        super(CancelModel, self).__init__()
        self.epoch = epoch

        # network parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if loss == 'BCE':
            self.loss = torch.nn.BCELoss()

        self.save_model_name = save_model_name # path + name


    def forward(self, x):
        #print(x)
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output
    
    
    def process_epoch(self, mode, epoch_idx, data_loader, log_interval=1000):
        epoch_loss = 0
        for batch_idx, batch in enumerate(data_loader):
            feature, label = batch['feature'], batch['target']
            if mode=='train':
                self.optimizer.zero_grad()
            
            predict_label = self.forward(feature)
            loss = self.loss(predict_label, label)
            loss_np = loss.data.numpy()
            epoch_loss += loss_np

            if mode=='train':
                loss.backward()
                self.optimizer.step()
            else:
                loss=None

            if batch_idx % log_interval == 0:
                print(mode.capitalize()+' Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                    epoch_idx, batch_idx , len(data_loader),
                    100. * batch_idx / len(data_loader), loss_np))
        print(mode.capitalize()+' set: Average loss: {:.4f}'.format(epoch_loss))
        return epoch_loss


    def train_model(self, train_data_loader, val_data_loader):
        train_loss = np.zeros(self.epoch)
        test_loss = np.zeros(self.epoch)
        for epoch_idx in range(1, self.epoch+1):
            self.train()
            train_loss[epoch_idx-1] = self.process_epoch('train', epoch_idx, train_data_loader)
            self.eval()
            test_loss[epoch_idx-1] = self.process_epoch('test', epoch_idx, val_data_loader)

            is_best = test_loss[epoch-1] < best_test_loss
            best_test_loss = min(test_loss[epoch-1], best_test_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'args': args,
                'state_dict': model.state_dict(),
                'best_test_loss': best_test_loss,
                'optimizer' : optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, self.save_model_name, is_best)
    

    def predict_model(self, test_data):
        feature, label = test_data['feature'], test_data['target']
        return self.forward(feature)
    

    def eval_model(self, test_data_loader):
        self.eval()
        right = 0
        total = len(test_data_loader)
        for batch_idx, batch in enumerate(test_data_loader):
            feature, label = batch['feature'], batch['target']
            predict_label = self.forward(batch)

            predict_label = (1 if predict_label[0] >= 0.5 else 0)
            if feature[0] == predict_label:
                right += 1
        print('Test Data Accuracy: {}/{}'.format(right, total))

    
    def save_model(self, save_model_path=''):
        save_model_name = self.save_model_name if save_model_path == '' else save_model_path
        if save_model_path == '':
            save_checkpoint({
                'state_dict': self.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
            }, save_model_name, is_best=False, is_final=True, is_training=False)

