#XOR implementation - forward pass
import numpy as np


def init_inputs():
    # x-no of inputs like x1,x2,x3...
    n_inputs = int(input("Enter the number of inputs: "))
    input_by_user=input(f"Do you want to enter inputs by yourself? Type 'yes' to confirm.").lower()

    if input_by_user == 'yes':
        x=[]
        for i in range(n_inputs):
            vals=float(input(f"Enter the inputs {i+1}:"))
            x.append(vals)
        x=np.array(x).reshape(n_inputs,1)
    else:
        x=np.random.rand(n_inputs,1)
    # l=hidden layers
    n_layers = int(input("Enter the number of hidden layers: "))
    layers = [n_inputs]  # adding the input layer to the layers list considering them as layer 0
    for i in range(n_layers):
        neurons = int(input(f"Enter the no of neurons in hidden layer {i + 1}:"))  # neurons inside each layer
        layers.append(neurons)
    n_outputs = int(input("Enter the no of neurons in the output layer: "))  #
    layers.append(n_outputs)  # adding the o/p layer as the final layer
    return x,layers
def init_weights_biases(layers):
    weights=[]
    biases=[]
    for i in range(len(layers)-1):
        # weights
        input_weights = input(f"Do you want to enter the weights for the layer {i+1} by yourself? Type yes to confirm.").lower()
        if input_weights == "yes":
            print(f"Enter the weights for layer {i+1} as space-separated values of shape ({layers[i+1]} x {layers[i]}):")
            values = list(map(float, input().split())) #for various inputs we convert all inputs to float
            w=np.array(values).reshape((layers[i+1],layers[i]))
        else:
            w=np.random.randn(layers[i+1],layers[i])
        weights.append(w)

        # biases
        input_biases = input(f"Do you want to enter the biases for the layer {i+1} by yourself? Type yes to confirm:").lower()
        if input_biases == "yes":
            print(f"Enter the biases for layer {i+1} as space-separated values of shape ({layers[i+1]} x 1):")
            b = list(map(float, input().split()))
            b = np.array(b).reshape((layers[i+1], 1))
        else:
            b=np.random.randn(layers[i+1],1)
        biases.append(b)
    return weights, biases

def activation_function(z):  # ReLU function is the activation function for the hidden layers
    return np.maximum(0, z)

def softmax(x):  # for the output layer
    s=np.exp(x) / np.sum(np.exp(x)) # exps = np.exp(x - np.max(x)) for stable softmax
    return s

def forward_pass(x,layers,weights,biases):
    a=x
    for i in range(len(layers) - 1):
        z=np.dot(weights[i],a)
        z += biases[i]
        a=activation_function(z)
        # if i<len(layers)- 2:
        #     a=activation_function(z)
        #     print(f"activation value after layer {i+1} is :\n{a}")
        # else:
        #     a=softmax(z)
    return a

def main():
    x,layers=init_inputs()
    weights,biases = init_weights_biases(layers)
    a=forward_pass(x,layers,weights,biases)
    print(f"The final output of the forward pass function is {a}")

if __name__=="__main__":
    main()
