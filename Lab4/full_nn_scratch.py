import numpy as np
import numpy as np
def init_inputs():
#x-no of inputs like x1,x2,x3...
    n_inputs=int(input("Enter the number of inputs: "))
    x=np.random.rand(n_inputs,1)
#l=hidden layers
    n_layers=int(input("Enter the number of hidden layers: "))
    layers=[n_inputs] #adding the input layer to the layers list considering them as layer 0
    for i in range(n_layers):
        neurons=int(input(f"Enter the no of neurons in layer {i+1}:")) #neurons inside each layer
        layers.append(neurons)
    n_outputs=int(input("Enter the no of neurons in the output layer: ")) #
    layers.append(n_outputs) #adding the o/p layer as the final layer
    return x,layers,n_layers

def init_weights(layers):
#weight matrix and biases matrix
    weights=[]
    biases=[]
    for i in range(len(layers)-1):
        w=np.random.randn(layers[i+1],layers[i])
        weights.append(w)
        b=np.random.randn(layers[i+1],1)
        biases.append(b)
    return weights,biases

def activation_function(z): #ReLU function is the activation function for the hidden layers
    return np.maximum(0,z)

def softmax(x): #for the output layer
    s = np.exp(x) / np.sum(np.exp(x))
    return s

def forward_pass(x,layers,weights,biases):
    a=x
    for i in range(len(layers)-1):
        z=np.dot(weights[i],a) + biases[i]
        if i< len(layers)-2:
            a=activation_function(z)
            print(f"activation value for layer {i+1} is {a}")
        else:
            a=softmax(z)
    return a
def main():
    x,layers,n_layers=init_inputs()
    weights,biases=init_weights(layers)
    a=forward_pass(x,layers,weights,biases)
    print(f"The output of the forward pass function is {a}")

if __name__=="__main__":
    main()
#Backpropagation from scratch (no hardcoded values)

import numpy as np
def sigmoid(x):
    return 1/(1 + np.exp(-x))
def sigmoid_derivative(output):
    return output*(1-output)
def init_input():
    np.random.seed(1)
    #4 layers in total, 2 hidden layers
    no_of_neurons=[2,4,3,1] #no of neurons in each layer
    #initializing inputs (x's)
    X=np.array([[2.0,3.0]]) #only hardcoded inputs and outputs
    y=np.array([[1.0]])

    #randomly initializing weights and biases -for all layers
    weights=[]
    biases=[]
    for i in range(len(no_of_neurons)-1):
        w=np.random.randn(no_of_neurons[i],no_of_neurons[i+1]) #matrix dim (no of neurons in the current layers no of neurons in the next layer)
        b=np.ones((1,no_of_neurons[i+1]))
        weights.append(w)
        biases.append(b)
    return X,y,weights,biases,no_of_neurons
def forward_propagate(X,y,weights,biases):
    a=X
    activations=[X] #storing the outputs of each layer to pass it on as input to the next layer
    z_values=[] #z=wx+b

    for w,b in zip(weights,biases):
        z=np.dot(a,w)+b
        z_values.append(z)
        a=sigmoid(z) #passing on z to the activation function
        activations.append(a)

    #computing the loss function
    loss=np.mean((activations[-1]-y)**2) #the last activation value is the y_pred activations[-1] MSE
    print(f"Loss function value: {loss:.4f}")
    return activations,z_values

def back_propagate(activations,y,weights,no_of_neurons,z_values):
    #gradient of loss func wrt loss function -dL/dL=1
    #gradient wrt output - dL/da=(a-y) -global gradient
    #dL/dz= da/dz.dL/da -local*global
    dL_dz=(activations[-1]-y)*sigmoid_derivative(activations[-1])

    #storing the gradients
    grad_w=[None]*len(weights) #empty lists to store gradients for wach layer
    # grad_b=[]*len(biases)

    #gradients of the last layer
    grad_w[-1]=np.dot(activations[-2].T,dL_dz)
    print(f"Gradient for L{len(weights)} (output layer):\n{grad_w[-1]}")

    #Backpropagate through the previous layers
    delta=dL_dz
    for l in range(2,len(no_of_neurons)):
        z=z_values[-l] #access the values in the list from the end (reverse)
        sp=sigmoid_derivative(activations[-l]) #derivative of the activations at each layer -local gradient
        delta=np.dot(delta,weights[-l+1].T)*sp #global * local
        grad_w[-l]=np.dot(activations[-l-1].T,delta) #computing weight gradients
        print(f"Gradient for L{len(weights)-l+1}:\n{grad_w[-l]}")
    return grad_w
def update(weights,grad_w,X,y,bias):
    alpha=0.1
    for i in range(len(weights)):
        weights[i] -= alpha*grad_w[i] #update the weights
    a=X
    for w,b in zip(weights,bias):
        a=sigmoid(np.dot(a,w)+b) #activation function
    new_loss=np.mean((a-y)**2)
    print(f"Loss function after updating: {new_loss:.4f}")

def main():
    X,y,weights,biases,no_of_neurons=init_input()
    activations,z_values=forward_propagate(X,y,weights,biases)
    grad_w=back_propagate(activations,y,weights,no_of_neurons,z_values)
    update(weights,grad_w,X,y,biases)
if __name__ == "__main__":
    main()
