import numpy as np

class Optimizer:
    def update(self,model,grad):
        raise NotImplementedError
    

class SGD(Optimizer):
    def __init__(self,lr=0.01):
        self.lr = lr
        
    
    def update(self,model,grad):
        "given wrights we have to update the weight"

        for i in range(model.num_layers):
            model.w[i] = model.w[i] - self.lr*grad['dw'+str(i+1)]
            model.b[i] = model.b[i]-self.lr*grad['db'+str(i+1)]


class MSGD(Optimizer):
    def __init__(self,lr=0.01,beta=0.9):
        self.lr  = lr
        self.beta=beta
        self.vb={}
        self.vw ={}

    def update(self, model, grad):
        for i in range(model.num_layers):
            key = i+1
            if key not in self.vw:
                self.vb[key] =0
                self.vw[key] =0

            self.vw[key] = self.beta*self.vw[key] + grad['dw'+str(i+1)]
            self.vb[key] = self.beta*self.vb[key] + grad['db'+str(i+1)]

            model.w[i] = model.w[i] - self.lr*self.vw[key]
            model.b[i] = model.b[i]-self.lr*self.vb[key]


class Nestrov(Optimizer):
    def __init__(self,lr=0.01,beta=0.9):
        self.lr  = lr
        self.beta=beta
        self.vb={}
        self.vw ={}

    def update(self, model, grad):
        for i in range(model.num_layers):
            key = i+1
            if key not in self.vw:
                self.vb[key] =0
                self.vw[key] =0

            vw_prev = self.self.vw[key]
            vb_prev = self.self.vb[key]

            self.vw[key] = self.beta*self.vw[key] + grad['dw'+str(key)]
            self.vb[key] = self.beta*self.vb[key] + grad['db'+str(i+1)]

            model.w[i] = model.w[i] - self.lr *(self.beta * vw_prev + grad['dw'+str(key)])
            model.b[i] = model.b[i] - self.lr*(vb_prev * self.beta + grad['db'+str(key)])


class RMSProp(Optimizer):
    def __init__(self,lr=0.01,beta=0.9,eps=1e-4):
            self.lr = lr
            self.beta = beta
            self.sw={}
            self.sb={}
            self.eps = eps

    def update(self, model, grad):
                for i in range(model.num_layers):
                    key = i+1
                    if key not in self.sw:
                        self.sb[key] =0
                        self.sw[key] =0

                    self.sw[key] = self.beta*self.sw[key] +(1- self.beta) * (grad['dw'+str(key)])**2
                    self.sb[key] = self.beta*self.sb[key] +(1- self.beta) * (grad['db'+str(key)])**2

                    model.w[i] -=  self.lr * self.grad['dw'+str(i+1)]/(np.sqrt( self.sw[key])+self.eps)
                    model.b[i] -= self.lr* self.grad['db'+str(i+1)]/(np.sqrt( self.sw[key])+self.eps)


                


class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.mW = {}
        self.vW = {}
        self.mb = {}
        self.vb = {}

        self.t = 0

    def update(self, model, grads):
        self.t += 1

        for i in range(model.num_layers):
            key = i + 1

            if key not in self.mW:
                self.mW[key] = 0
                self.vW[key] = 0
                self.mb[key] = 0
                self.vb[key] = 0

            # Momentum
            self.mW[key] = self.beta1 * self.mW[key] + (1 - self.beta1) * grads["dw" + str(key)]
            self.mb[key] = self.beta1 * self.mb[key] + (1 - self.beta1) * grads["db" + str(key)]

            # RMSProp
            self.vW[key] = self.beta2 * self.vW[key] + (1 - self.beta2) * (grads["dw" + str(key)] ** 2)
            self.vb[key] = self.beta2 * self.vb[key] + (1 - self.beta2) * (grads["db" + str(key)] ** 2)

            # Bias correction
            mW_hat = self.mW[key] / (1 - self.beta1 ** self.t)
            mb_hat = self.mb[key] / (1 - self.beta1 ** self.t)
            vW_hat = self.vW[key] / (1 - self.beta2 ** self.t)
            vb_hat = self.vb[key] / (1 - self.beta2 ** self.t)

            # Update
            model.w[i] -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
            model.b[i]  -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)



  
                      



        

        





