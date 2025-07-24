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






