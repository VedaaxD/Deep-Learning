#Implementation of batch normalization from scratch and layer normalization for training deep networks.

import numpy as np

class Batchnorm():
    def __init__(self,num_features,epsilon=1e-5,momentum=0.9,lr=0.01):
        self.gamma=np.ones((1,num_features)) #scale parameter
        self.beta=np.ones((1,num_features)) #shift parameter
        self.epsilon=epsilon
        self.lr=lr
        # self.momentum=momentum #for moving averages , needn't use momentum for now
        # self.running_mean=np.zeros((1,num_features))
        # self.running_var=np.ones((1,num_features))
    def forward(self,x):
        self.x=x
        self.batch_mean=np.mean(self.x,axis=0,keepdims=True) #mean per feature
        self.batch_var=np.var(x,axis=0,keepdims=True) #variance per feature
        self.z_hat=(x-self.batch_mean)/np.sqrt(self.batch_var+self.epsilon) #normalization
        out=self.gamma*self.z_hat +self.beta # (z=gamma(z)+beta) - scale and shift
        # #update the running statistics
        # self.running_mean=(1-self.momentum)*self.running_mean + self.momentum*self.batch_mean
        # self.running_var=(1-self.momentum)*self.running_var + self.momentum*self.batch_var
        # #formula => (1-m)*running_mean + m * current_mean
        return out
    def backward(self,grad_ops):
        N=self.x.shape[0]
        dL_dgamma=np.sum(grad_ops*self.z_hat,axis=0,keepdims=True) #dL/dgamma=dL/dy.x
        dL_dbeta=np.sum(grad_ops,axis=0,keepdims=True) #dL/dbeta = dL/dy

        #update params
        self.gamma-=self.lr *dL_dgamma
        self.beta-=self.lr *dL_dbeta
        #gradient wrt input
        dL_dzhat=grad_ops*self.gamma #dL/dzhat=dL/dy * gamma

        #gradient wrt variance
        dL_dvar=np.sum(dL_dzhat*(self.x-self.batch_mean)*-0.5*(self.batch_var+self.epsilon)**(-1.5),axis=0,keepdims=True)
        #dL/dsigma^2= dL/dzhat . (x- mean).(0.5).(sigma^2+eps)^3/2

        #gradient wrt mean
        dL_dmean=np.sum(dL_dzhat* -1/np.sqrt(self.batch_var+self.epsilon),axis=0,keepdims=True)+ dL_dvar * np.mean(-2*(self.x-self.batch_mean),axis=0,keepdims=True)
        #dL_dmean= (dL_dzhat . -1/sqrt(sigma^2+eps)) + dL_dsigma^2 . -2/N(x-mean)

        #gradient wrt input x
        dx=dL_dzhat/np.sqrt(self.batch_var+self.epsilon)+dL_dvar*2*(self.x-self.batch_mean)/N+dL_dmean/N
        return dx
        #full gradient is
        #dL/dx = dL/dzhat . 1/sqrt(sigma^2+eps) + dL/dsigma^2 .2(x-mean)/N +dL/dmean . 1/N

class LayerNorm():
    #layernorm normalizes per sample , across features
    def __init__(self,num_features,epsilon=1e-5,lr=0.01):
        self.gamma=np.ones((1,num_features))
        self.beta=np.zeros((1,num_features))
        self.epsilon=epsilon
        self.lr=lr

    def forward(self,x):
        self.x=x
        self.mean=np.mean(x,axis=1,keepdims=True) #note: per sample meana nd variance
        self.var=np.var(x,axis=1,keepdims=True)
        self.z_hat=(x-self.mean)/np.sqrt(self.var+self.epsilon)
        out=self.gamma*self.z_hat+self.beta
        return out

    def backward(self,grad_ops):
        N,D=self.x.shape
        #Gradients wrt gamma and beta
        d_gamma=np.sum(grad_ops*self.z_hat,axis=0,keepdims=True)
        d_beta=np.sum(grad_ops,axis=0,keepdims=True)

        #Update params
        self.gamma-=self.lr*d_gamma
        self.beta-=self.lr*d_beta
        #Gradient wrt input
        dL_dzhat=grad_ops*self.gamma

        dL_dvar=np.sum(dL_dzhat*(self.x-self.mean)*-0.5*(self.var+self.epsilon)**(-1.5),axis=1,keepdims=True)

        dL_dmean=np.sum(dL_dzhat * -1 / np.sqrt(self.var+self.epsilon),axis=1,keepdims=True)+\
                        dL_dvar*np.mean(-2*(self.x-self.mean),axis=1,keepdims=True)

        dx=dL_dzhat/np.sqrt(self.var+self.epsilon)+ dL_dvar*2*(self.x-self.mean)/D+dL_dmean/D

        return dx
#we'll need to check whether BatchNorm normalizes each feature in fwd pass, also compare the analytical gradient
#from bwd pass with a numerical gradient
def test_batchnorm_fwd_bwd():
    np.random.seed(42)
    N,D=4,3 #N- no of samples (rows), D- features per sample (columns)
    X=np.random.rand(N,D)*5 +10 #eandom batch initialization with mean=10 and var=25
    bn=Batchnorm(num_features=D,lr=0.01)

    #fwd test
    out=bn.forward(X)
    mean=np.mean(out,axis=0)
    var=np.var(out,axis=0)
    print(f"Forward test")
    print(f"Mean per feature (should be tending to 0):{mean}")
    print(f"Variance per feature (should be tending to 1):{var}")

    #bwd test
    grad_out=np.random.randn(*out.shape) #random gradient from the next layer
    dx=bn.backward(grad_out)

    #numerical gradient check
    eps=1e-5
    grad_num=np.zeros_like(X)
    for i in range(N):
        for j in range(D):
            X_pos=X.copy();X_pos[i,j]+=eps
            X_neg=X.copy();X_neg[i,j]-=eps
            out_pos=bn.forward(X_pos)
            out_neg=bn.forward(X_neg)
            #out_pos-out_neg -how much the o/ps changed when the i/ps was slightly nudged with epsilon
            grad_num[i,j]=np.sum((out_pos-out_neg)*grad_out)/(2*eps)
            #f(x)= f(x+e) - f(x-e)/2e

    #Relative error
    rel_error=np.linalg.norm(dx - grad_num)/(np.linalg.norm(dx) +np.linalg.norm(grad_num))
    print(f"\nBackward Test")
    print(f"Relative error should be very less (<1e-6): {rel_error}")



#testing batchnorm with a toy dataset
X=np.array([[1.0,2.0,3.0],
           [2.0,3.0,4.0],
           [3.0,4.0,5.0]])
#dummy gradients
#batchnorm test
grad=np.random.randn(*X.shape) #which should be the same shape as X
bn=Batchnorm(num_features=3,lr=0.01)
out=bn.forward(X)
dx=bn.backward(grad)
print(f"Forward Output:\n{out}")
print(f"Gradient wrt input dx:\n{dx}")
print(f"Updated gamma:\n{bn.gamma}")
print(f"Updated beta:\n{bn.beta}")

#layernorm test
ln = LayerNorm(num_features=3, lr=0.01)
out_ln =ln.forward(X)
dx_ln =ln.backward(grad)
print(f"\n[LayerNorm]")
print(f"Forward Output:\n{out_ln}")
print(f"Gradient wrt input dx:\n{dx_ln}")
print(f"Updated gamma:\n{ln.gamma}")
print(f"Updated beta:\n{ln.beta}")

test_batchnorm_fwd_bwd()