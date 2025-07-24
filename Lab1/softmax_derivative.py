#softmax derivative
import numpy as np

def softmax_function(z):
    s=np.exp(z)/np.sum(np.exp(z))
    return s

def softmax_derivative(s):
    jacobian=[]
    n=len(s)
    for i in range(n):
        row=[]
        for j in range(n):
            if i==j:
                val=s[i]*(1-s[i])
            else:
                val=-s[i]*s[i]
            row.append(val)
        jacobian.append(row)
    return jacobian
def main():
    z=[2,1,0.1]
    s=softmax_function(z)
    print(f"Softmax function applied {(s)}")
    j=softmax_derivative(s)
    print("Jacobian matrix:")
    for row in j:
        print(row)
if __name__=="__main__":
    main()



