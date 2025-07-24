import numpy as np
import matplotlib.pyplot as plt
#various activation function
x=np.linspace(0,1,100)
def sigmoid(x):
    return 1/ 1+ np.exp(-x)

def sigmoid_derivative(x):
    s=sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def tanh_derivative(x):
    return 4/(np.exp(x)+np.exp(-x))**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x>0,1,0)

def leaky_relu(x, alpha=0.01):
    return np.where(x>0,x,alpha*x)

def leaky_relu_derivative(x,alpha=0.01):
    return np.where(x>0,1,alpha)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x)) #didn't normalise as the range is from -10 to 10

def softmax_derivative(x):
    s=softmax(x)
    return np.diag(s)-np.outer(s,s) #np.diag, gives diagonal matrix, where all entries are 0 except the diagonal
                                    #np.outer gives the matrix, where si.sj, by subtracting it we arrive at Jacobian matrix.

def plots(name,func,derivative,z):
    y=func(z)
    dy=derivative(z)
    plt.figure(figsize=(8,8))
    plt.plot(z,y,label=f"{name}")
    plt.plot(z,dy,label=f"{name} Derivative",linestyle="--")
    plt.title(f"{name} and its derivative")
    plt.xlabel("z")
    plt.ylabel("value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    x=np.linspace(-10,10,100)
    plots("sigmoid",sigmoid,sigmoid_derivative,x)
    plots("tanh",tanh,tanh_derivative,x)
    plots("relu",relu,relu_derivative,x)
    plots("leaky relu",leaky_relu,relu_derivative,x)

    #we need vector values for the softmax function input
    z=np.linspace(-10,10,100)
    y=softmax(z)
    dy=softmax_derivative(z)

    plt.figure(figsize=(8,5))
    plt.plot(z,y,label="softmax")
    plt.title("Softmax")
    plt.xlabel("Z")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() #this will plot softmax function
    # #softmax derivative
    # plt.plot(np.diag(dy),label="Softmax derivative function")
    # plt.title("Jacobian of Softmax derivative function")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()





    