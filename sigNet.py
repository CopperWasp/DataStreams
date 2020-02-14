import torch
from torch.autograd import Variable
import parameters as p
import numpy as np
import math



torch.manual_seed(p.random_seed)
LR = p.signet_lr
# weights are in the format of: (hidden, input), (output, hidden)
# when expanding input, you need to add new weights columnwise, since each column represents a feature
# to get parameters: layer.weight.data.numpy()
# self.conv1.weight = torch.nn.Parameter(K)

class sigNet(torch.nn.Module):
    def __init__(self, input_size):
        super(sigNet, self).__init__()
        torch.manual_seed(p.random_seed)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.input_hidden = torch.nn.Linear(input_size, 1, bias=False)
        torch.nn.init.normal(self.input_hidden.weight, 0, 0.00001)

        
    def pruneWeights(self, multiplier):
        for child in self.children():
            for param in child.parameters():
                #print(np.array(multiplier).shape)
                param = (param * torch.FloatTensor(np.array(multiplier)))
    
        
    def forward(self, x):
        self.grow(x)
        self.feature_relevances = self.featureSelection()
        x1 = self.sigmoid(self.input_hidden(x))
        return x1
    
    
    def getPrediction(self, x):
        x = Variable(torch.FloatTensor(x))
       # if math.isnan(np.array(((self.forward(x)).detach()))[0]):
        #    print("Network")
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
        
        return feature_relevances
    
class signetTrainer():
    def __init__(self, input_size):
        self.model = sigNet(input_size)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LR, weight_decay=p.signet_l2)
        self.criterion = torch.nn.KLDivLoss(size_average=False)
        
    def trainSigNet(self, x, y, i, num_new):
        x = Variable(torch.FloatTensor(x)) # convert into 0-1 activations
        y = Variable(torch.FloatTensor(np.array(y)))
        output = self.model.forward(x)
        
        loss = self.criterion(output, y)
        loss.backward()
        #print(loss)
       # print(grad_params)
        grad_norm = 0
        i= 0 
        for pa in self.model.parameters(): # existing ones
            #print(grad)
            if(i<len(list(self.model.parameters()))-num_new):
                #grad_norm += grad.pow(2).sum()
                pa.grad*=p.grad_norm_multiplier
            else:
                pa.grad*=p.new_norm_multiplier
                
            i+=1
        if grad_norm != 0:
            grad_norm = grad_norm.sqrt()
            
            
            
        params = list(self.model.parameters())
        para_norm = 0
        for i in range(len(params)):
            if(i>=len(params)-num_new):
                para_norm += params[i].pow(2).sum()
        if para_norm != 0:
            para_norm = para_norm.sqrt()
        
        #print(grad_norm)
        #print(p.grad_norm_multiplier, p.new_norm_multiplier)
        #loss = self.criterion(output, y)# + p.accumulator * (grad_norm * (p.grad_norm_multiplier) + p.new_norm_multiplier * para_norm)
         # gradient
        
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss
        #print(self.model.input_hidden)
    
    

    
    
            
        
   
