#This is a scratch code of the forward pass- network 1 with one layer
import numpy as np

def input_vector():
    z=np.random.rand(4,1) #input layer
    weights=np.array([[0.1,0.1,0.1,0.1]])
    r=np.dot(weights,z) #resultant array
    return r
def softmax_function(r):
    s=np.exp(r)/np.sum(np.exp(r))
    return s
def main():
    r=input_vector()
    s = softmax_function(r)
    print(f"The output of this forward pass is:{s}")
if __name__=='__main__':
    main()

