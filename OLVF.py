import numpy as np
import preprocess
import random
import copy
import parameters as p
import matplotlib.pyplot as plt
from sklearn import linear_model



class olvf:
    def __init__(self, data):
        self.C=p.olvf_C
        self.B=p.olvf_B
        self.Lambda = 30
        self.data=data
        self.rounds = p.cross_validation_folds
        self.error_stats = []
        self.sparse_multiplier = [1] * (len(self.data[0])-1) # exclude label
        

    def sparsity_step(self):
        #projected= np.multiply(np.minimum(1, self.Lambda/np.linalg.norm(self.weights, ord=1)), self.weights)
        self.weights= self.truncate(self.weights)
        
        
    def truncate(self, projected):
        if np.linalg.norm(projected, ord=0)> self.B*len(projected):
            remaining= int(np.maximum(1, np.floor(self.B*len(projected))))
            for i in projected.argsort()[:(len(projected)-remaining)]:
                projected[i]=0
            return projected
        else: return projected
        
        
    def set_classifier(self):
        self.weights = np.random.randn(np.count_nonzero(self.X[0])) * 0.001
        #self.weights = np.zeros(len(self.X[0]))
        self.reg_weights = np.random.randn(np.count_nonzero(self.X[0])) * 0.001


    def netSparsity(self, relevances, sparse_scalar, nonz_length):
        # keeps track of self.sparsity_multiplier
        # with each new feature, there will we more features to remove
        allowed = int(nonz_length * sparse_scalar)
        remove = len(self.weights) - allowed
        rel = np.absolute(relevances[:len(self.weights)])
        remove_indices = np.argsort(rel)[:remove]
        for i in range(len(self.weights)):
            if i in remove_indices:
                self.sparse_multiplier[i] = 0


    
    def mseLoss(self, x, y, w):
        return (np.square((y - self.sigmoid(np.dot(w, x))), dtype=np.float64) * 0.5)
    
    
    def mseLossDerivative(self, x, y, w):
        act = self.sigmoid(np.dot(w,x))
        return (np.maximum(np.multiply((np.dot(w, x) - y), x) * act * (1-act) - p.eps, 0))+ p.l2 * w

    
    def sigmoid(self, x):        
        sig = np.divide(1.0,(1.0 + np.exp(-x)))
        return sig
    
    
    def crossEntropyLoss(self, x, y, w):
        y_hat = self.sigmoid(np.dot(w, x))
        loss = np.log(1.0 + np.exp(-(y_hat * y)))
        return loss
    
    
    def crossEntropyLossDerivative(self, x, y, w):     
        #exp_term = np.exp(-(y * np.dot(w, x)))
        #dl = np.multiply(-(np.divide(exp_term,(exp_term + 1.0))), np.multiply(y, x))
        dl = np.multiply(x, (self.sigmoid(np.dot(w, x)) - y))
        return dl


    def fit(self):
        #np.seterr(all='raise')
        random.seed(p.random_seed)
        print("OLVF-closedForm Cross-validation Round (" +str(self.rounds) +")")
        for k in range(0, self.rounds):
            if k%2 == 0:
                print(str(k)+" ", end='\r')
            self.getShuffledData() # each round get shuffled data from same seed as other algorithm
            train_error = 0
            conf_loss = 0
            train_error_vector=[]
            total_error_vector= np.zeros(len(self.y))
            
            self.set_classifier()
            total_reg_loss = 0
            
            

            for i in range(0, len(self.y)):
                row = np.array(self.X[i][:np.count_nonzero(self.X[i])])
                reg_row = np.array([1 if i!=0 else 0 for i in row])
                nonz_length = np.count_nonzero(self.X[i])
                
                
                #row *= self.sparse_multiplier[:len(row)] # mask for sparsity
                
                if len(row)==0: continue
            

                #y_hat_reg = self.sigmoid(np.dot(self.reg_weights, row[:len(self.reg_weights)]))  
                y_hat= np.sign(np.dot(self.weights, row[:len(self.weights)]))
                if y_hat!=self.y[i]: 
                    train_error+=1
                    reg_label = 0
                else:
                    reg_label = 1
                    
                loss= (np.maximum(0, 1-self.y[i]*(np.dot(self.weights, row[:len(self.weights)]))))


                
                # logistic regressor update
                reg_loss = float(self.crossEntropyLoss(reg_row[:len(self.reg_weights)], reg_label, self.reg_weights))
                reg_loss2 = float(self.crossEntropyLoss(reg_row[len(self.reg_weights):], reg_label, np.zeros(len(reg_row[len(self.reg_weights):]))))
                reg_loss_prime = self.crossEntropyLossDerivative(reg_row[:len(self.reg_weights)], reg_label, self.reg_weights) 
                reg_loss_prime2 = self.crossEntropyLossDerivative(reg_row[len(self.reg_weights):], reg_label, np.zeros(len(reg_row[len(self.weights):])))
               
                reg_w_1 = self.reg_weights + np.multiply(min(p.olvf_C2, reg_loss), np.divide(1.0, (reg_loss_prime+0.0001)))
                reg_w_2 = np.multiply(min(self.C, reg_loss2), np.divide(1.0, (reg_loss_prime2+0.0001)))
                self.reg_weights = np.append(reg_w_1, reg_w_2)
            
                # regressor prediction
                exist = [1] * nonz_length + (len(self.reg_weights) - nonz_length) * [0]#.reshape(1, -1)
                new = (np.array([0] * nonz_length + (len(self.reg_weights) - nonz_length) * [1]))#.reshape(1, -1)
                sc_w_1 = self.sigmoid(np.dot(self.reg_weights, exist[:]))#clf.predict(exist)#np.dot(self.reg_weights, exist)
                sc_w_2 = self.sigmoid(np.dot(self.reg_weights, new))#clf.predict(new)#np.dot(self.reg_weights, new)
            
                #print(sc_w_1, sc_w_2)
                pw1 = np.round(sc_w_1, 2)
                pw2 = np.round(sc_w_2, 2)
                print(pw1, pw2)


                # clfr  update
                tao= np.minimum(self.C, loss/np.dot(self.X[i], self.X[i]))
                w_1= self.weights+ pw1 * np.multiply(tao*self.y[i], row[:len(self.weights)])
                w_2= pw2 * np.multiply(tao*self.y[i], row[len(self.weights):])
                self.weights= np.append(w_1, w_2)
                #self.sparsity_step()
                

                
                
                
                #self.netSparsity(self.reg_weights, p.olvf_B, nonz_length)
                #self.weights = np.multiply(self.weights, self.sparse_multiplier[:len(self.weights)])
                #self.reg_weights= np.multiply(self.reg_weights, self.sparse_multiplier[:len(self.reg_weights)])
                #print(i, self.sparse_multiplier)
                
                #print("weights and mult at round: "+str(i))
                #print(self.sparse_multiplier)
                #print(self.weights)
                #print(self.sparse_multiplier)
                
                train_error_vector.append(train_error/(i+1))
                
                # real time plot
                #plt.scatter(i, (total_reg_loss / (float(i)+1)), color='red')
                #plt.scatter(i, (conf_loss.detach().numpy() / (float(i)+1)), color='blue')
                #plt.scatter(i, np.linalg.norm(1.0/ loss_prime), color = 'red')
                #plt.scatter(i, np.linalg.norm(self.weights), color='blue')
                #plt.pause(0.00001) 
                # real time plot
                #print(i, loss, self.reg_weights)
                
                
                
            plt.show()
            
            total_error_vector= np.add(train_error_vector, total_error_vector)
            self.error_stats.append(train_error)
        total_error_vector= np.divide(total_error_vector, self.rounds)

        print("olvf avg. error: "+str(train_error_vector[-1]))
        print(np.count_nonzero(self.weights))
        return train_error_vector
                
    
    def predict(self, X_test):
        prediction_results=np.zeros(len(X_test))
        for i in range (0, len(X_test)):
            row= X_test[i]
            prediction_results[i]= np.sign(np.dot(self.weights, row[:len(self.weights)]))
        return prediction_results


    def getShuffledData(self): # generate data for cross validation
        data=self.data
        copydata = copy.deepcopy(data)
        random.Random(p.random_seed).shuffle(copydata)
        dataset=preprocess.removeDataTrapezoidal(copydata)
        all_keys = set().union(*(d.keys() for d in dataset))
    
        X,y = [],[]
        for row in dataset:
            for key in all_keys:
                if key not in row.keys() : row[key]=0
            y.append(row['class_label'])
            del row['class_label']
        if 0 not in row.keys(): start=1
        if 0 in row.keys(): start=0
        for row in dataset:
            X_row=[]
            for i in range(start, len(row)):
                X_row.append(row[i])
            X.append(X_row)
        self.X, self.y = X, y           
        

