#Implement dropout to regularize neural networks from scratch.
import numpy as np
class Dropout():
    #we generate a random number for every entry in the input matrix X. then for each entry, if the random number > dropout prob
    #We keep that input (mask=1) ,else we drop it (mask=0)
    #dropout(x)=m.x/(1-p) dot is element-wise multiply, where m is random masks of 0s and 1s sampled from Bernoulli(1-p)
    def __init__(self,p=0.5):
        self.p=p #probability of dropping a neuron
        self.mask=None
        self.training=True #flag to differentiate train/test
    def forward(self,X):
        if self.training:
            #Sample a mask of 0s and 1s 0-drop, 1-keep
            self.mask=(np.random.randn(*X.shape)>self.p).astype(np.float32)
            out=(X*self.mask)/(1-self.p) #scale the kept neurons so expect values will stay the same
        else:
            # in the inference phase, as the dropouts will not be used
            out=X
        return out
    def backward(self,grad_output):
        #Gradient will flow through only the kept neurons
        dx=(grad_output*self.mask)/(1-self.p)
        return dx

np.random.seed(0)
X=np.array([[1.0,2.0,3.0],
            [2.5,7.7,6.4],
            [5.8,6.0,9.2]])
#model
drop=Dropout(p=0.5)
#training
#forward pass
out_train=drop.forward(X)
print(f"Train forward:\n{out_train}")

#backward pass
grad_out=np.ones_like(X)
dx=drop.backward(grad_out)
print(f"Backward gradient:\n{dx}")

#evaluation
drop.training=False
out_test=drop.forward(X)
print(f"Test forward:\n{out_test}")

