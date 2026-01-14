import numpy as np

class NeuralNetwork():
    def __init__(self,layer_sizes):
        
        self.layer_size = layer_sizes
        self.num_layers = len(self.layer_size)-1

        self.w =[]
        self.b=[]
        for i in range(self.num_layers):
            W =np.random.rand(self.layer_size[i],self.layer_size[i+1])*0.1
            b = np.zeros((1,self.layer_size[i+1]))
            self.w.append(W)
            self.b.append(b)

        
    def relu(self,x):
        return np.maximum(0,x)
    
    def softmax(self,x):
        z = x - np.max(x,axis=1,keepdims=True)
        exp_z = np.exp(z)

        return(exp_z/np.sum(exp_z,axis=1,keepdims=True))
    
    def forward(self,x):
        self.cache = {}
        a = x
        self.cache['a0'] = a
        for i in range(self.num_layers-1):
            z = np.dot(a,self.w[i])+self.b[i]
            a = self.relu(z)
            self.cache[f'z{i+1}'] = z
            self.cache[f'a{i+1}']=a

        zL =np.dot(a,self.w[-1])+self.b[-1]
        y_hat = self.softmax(zL)
        self.cache['zL'] =zL
        self.cache['aL']=y_hat
        return y_hat
    

    def backward(self,y_true):
        grads ={}
        m = y_true.shape[0]

        dz = self.cache['aL'].copy()
        dz[np.arange(m),y_true] -=1
        
        dz /=m

        a_prev = self.cache[f"a{self.num_layers-1}"]
        # grads["dw"+ str{self.num_layers}] = np.dot(a_prev.T,dz)
        grads["dw" + str(self.num_layers)] = np.dot(a_prev.T, dz)
        grads["db" + str(self.num_layers)] = np.sum(dz, axis=0, keepdims=True)

        da = np.dot(dz, self.w[-1].T)

        for i in reversed(range(self.num_layers - 1)):

                z = self.cache[f"z{i+1}"]
                dz = da * (z > 0)
                a_prev = self.cache[f'a{i}']

                grads['dw' + str(i+1)] = np.dot(a_prev.T,dz)
                grads["db" + str(i+1)] = np.sum(dz,axis=0,keepdims=True)

                da = np.dot(dz,self.w[i].T)
        return grads
    
    def evaluate(self,y_pred,y_true):
        pred = np.argmax(y_pred,axis=1)
        return np.mean(pred==y_true)










    
    


    
        


    


