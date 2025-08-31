#Write a program to simulate vanishing & exploding gradient problems.
import torch
import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score


torch.manual_seed(42)
np.random.seed(42)

#Data
X,y=make_circles(n_samples=1000,factor=0.3,noise=0.1)
X=torch.tensor(X,dtype=torch.float32)
y=torch.tensor(y,dtype=torch.float32).unsqueeze(1) #dimension-(1000,1)
plt.figure(figsize=(8,6))
plt.scatter(X[:,0],X[:,1],c=y.squeeze())
plt.show()

#Define MLP with ReLU
class MLP(nn.Module):
    def __init__(self,activation=nn.ReLU()):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(2,5),
            activation,
            nn.Linear(5,3),
            activation,
            nn.Linear(3,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.net(x)

#defining the model
model=MLP(activation=nn.ReLU())
optimizer=optim.Adam(model.parameters(),lr=0.01)
criterion=nn.BCELoss()

#training loop
weights=[]
gradients=[]

for epoch in range(100):
    y_pred=model(X)
    loss=criterion(y_pred,y)
    optimizer.zero_grad()
    loss.backward()
    # capturing the weight stats
    # p.data-gets the tensor data for that parameter
    # .clone() makes a copy so updates don't get overwritten
    # .detach() removes the tensor from the pytorch's autograd graph- since we don't want gradient tracking
    # .numpy()-tensor to numpy

    snapshot_w = {name: p.data.clone().detach().numpy() for name, p in model.named_parameters()
                  if "weight" in name}
    weights.append(snapshot_w)

    # capturing the gradient stats
    snapshot_g = {name: p.grad.clone().detach().numpy() for name, p in model.named_parameters()
                  if "weight" in name}
    gradients.append(snapshot_g)
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss:{loss.item():4f}")
print(f"Final accuracy using ReLU as activation function"
      f":{accuracy_score(y.detach().numpy(),(y_pred.detach().numpy()>0.5).astype(int)):.4f}")

#vanishing gradient problem
#using sigmoid function as activation function throughout
#using deeper networks - 8 layers
class DeepMLP(nn.Module):
    def __init__(self,activation=nn.Sigmoid()):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(2,5),activation,
            nn.Linear(5,7),activation,
            nn.Linear(7,4),activation,
            nn.Linear(4,5),activation,
            nn.Linear(5,8),activation,
            nn.Linear(8,3),activation,
            nn.Linear(3,3),activation,
            nn.Linear(3,1),activation,

        )
    def forward(self,x):
        return self.net(x)

#defining the model
model=DeepMLP(activation=nn.Sigmoid())
optimizer=optim.SGD(model.parameters(),lr=0.01)
criterion=nn.BCELoss()

#training loop
#also we are learning how to capture weights and gradients
weights=[]
gradients=[]

for epoch in range(100):
    y_pred=model(X)
    loss=criterion(y_pred,y)
    optimizer.zero_grad()
    loss.backward()

    #capturing the weight stats
    #p.data-gets the tensor data for that parameter
    #.clone() makes a copy so updates don't get overwritten
    #.detach() removes the tensor from the pytorch's autograd graph- since we don't want gradient tracking
    #.numpy()-tensor to numpy

    snapshot_w={name:p.data.clone().detach().numpy() for name,p in model.named_parameters()
                if "weight" in name}
    weights.append(snapshot_w)

    #capturing the gradient stats
    snapshot_g={name:p.grad.clone().detach().numpy() for name,p in model.named_parameters()
                if "weight" in name}
    gradients.append(snapshot_g)
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch:{epoch}, Loss={loss.item():.4f}")

acc=accuracy_score(y.detach().numpy(),(y_pred.detach().numpy()>0.5).astype(int))
print(f"Final accuracy using deepMLP and sigmoid as activation:{acc:.4f}")
#Glorot initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight) #Glorot
        nn.init.zeros_(m.bias)
model=DeepMLP(activation=nn.Sigmoid())
model.apply(init_weights)


def plot_means_and_std(history, title):
    for key in history[0]:
        means = [h[key].mean() for h in history]
        stds = [h[key].std() for h in history]

        plt.plot(means, label=f"{key} mean")
        plt.plot(stds, '--', label=f"{key} std")  # dashed line for std

    plt.title(f"{title} mean & std across epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend(loc="best")
    plt.show()

plot_means_and_std(weights,"Weights")
plot_means_and_std(gradients,"Gradients")


