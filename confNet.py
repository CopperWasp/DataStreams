import torch
from torch.autograd import Variable
import parameters as p
import numpy as np
import math


torch.manual_seed(p.random_seed)

LR = p.confnet_lr
l2 = p.confnet_l2
# weights are in the format of: (hidden, input), (output, hidden)
# when expanding input, you need to add new weights columnwise, since each column represents a feature
# to get parameters: layer.weight.data.numpy()
# self.conv1.weight = torch.nn.Parameter(K)

class confidenceNet(torch.nn.Module):
    def __init__(self, input_size):
        super(confidenceNet, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.input_hidden = torch.nn.Linear(input_size, 1, bias=False)
        #self.input_hidden2 = torch.nn.Linear(4, 1, bias=False)
        #self.hidden_output = torch.nn.Linear(p.hidden_size, 1, bias=False)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        #print(self.parameters())
        torch.nn.init.normal(self.input_hidden.weight, 0, 1)
        #torch.nn.init.normal(self.input_hidden2.weight, 0, 1)
        
        
    def forward(self, x):
        self.grow(x)
        self.feature_relevances = self.featureSelection()
        x1 = self.sigmoid(self.input_hidden(x))
        #x2 = self.sigmoid(self.input_hidden2(x1))
        #x3 = (self.hidden_output(x1))
        return x1
    
    
    def getPrediction(self, x):
        x = Variable(torch.FloatTensor([1 if i!=0 else i for i in x]))
        #print(np.array(self.sigmoid(self.forward(x)).detach())[0])
        if math.isnan(np.array(((self.forward(x)).detach()))[0]):
            print("Network")
        return np.array(((self.forward(x)).detach()))[0]
    
    
    def grow(self, x):
        input_size = self.input_hidden.weight.data.shape[1]
        hidden_size = self.input_hidden.weight.data.shape[0]
        difference = len(x) - input_size
        if(difference <= 0): return
        current_wts = self.input_hidden.weight.data.numpy()
        # generate a matrix of dimensions (hidden_size, difference), and columnwise append to current weights
        addition = np.random.randn(hidden_size, difference) 
        #addition = np.ones((hidden_size, difference))
        current_wts = np.append(current_wts, addition, axis=1) * 0.0001
        self.input_hidden = torch.nn.Linear(current_wts.shape[1], current_wts.shape[0], bias=False)
        self.input_hidden.weight = torch.nn.Parameter(torch.from_numpy(current_wts).float())
        #print(list(self.parameters()))
        
    
    def featureSelection(self):
        in_matrix = (self.input_hidden.weight.data.detach().numpy())
        feature_relevances = []
        for i in range(in_matrix.shape[1]):
            feature_relevances.append((np.average((np.absolute((in_matrix[:, i]))))))
        #print(feature_relevances)
        #print(list(self.parameters())[0].grad)
        #print(list(self.parameters())[1].grad)
        #print(list(self.parameters())[2].grad)
        #print(feature_relevances)
        #rint(self.input_hidden.weight.data)
        return feature_relevances
    
class confnetTrainer():
    def __init__(self, input_size):
        self.model = confidenceNet(input_size)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LR, weight_decay=l2)
        self.criterion = torch.nn.MSELoss(size_average=False)
        
    def trainConfNet(self, x, y):
        x = Variable(torch.FloatTensor([1 if i!=0 else i for i in x])) # convert into 0-1 activations
        y = Variable(torch.FloatTensor(np.array(y)))
        output = self.model.forward(x)
        loss = self.criterion(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
        #print(self.model.input_hidden)
    
    

    
    
            
        
   
