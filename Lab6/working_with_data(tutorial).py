#following the tutorial..
#Quickstart part of the tutorial
#Working with data
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data", #Folder to store the data
    train=True,
    download=False, #already downloaded in terminal
    transform=ToTensor(), #converts to tensor format
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False, #already downloaded in terminal
    transform=ToTensor(),
)
#Batching with dataloaders
batch_size = 64
#Dataloaders splits the data into mini-batches of size 64
train_dataloader =DataLoader(training_data,batch_size=batch_size)
test_dataloader =DataLoader(test_data,batch_size=batch_size)

#viewing the sample images
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}") #N-batch size,C,H,W -tensor variables
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols,rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows,cols,i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

#setting the device - cpu or gpu
device= torch.accelerator.current.accelerator().type if torch.accelerator.is_available() else "cpu" #checks if any compatible accelerators are available
print(f"Using {device} device")

#Defining the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() #converts each 2d image(28*28) into a flat vector(784)
        self.linear_relu_stack=nn.Sequential( #sequentially arranges hidden layers -a chain of layers
            nn.Linear(28*28,512), #z- multiplies the inputs and weights
            nn.ReLU(), #Activation function is applied
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10), #output layer containing 10 o/ps
        )
    def forward(self,x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x) #raw prediction scores w/o softmax by applying the above methods
        return logits
model=NeuralNetwork().to(device) #instance of the model and moves it to cpu
print(model)

#for training the model we need loss function and optimizer
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=1e-3) #stochastic gd optimizer

#the data is fed in batches to the model, and predictions are made
#backpropagates the prediction error to update/adjust the model parameters

def train(dataloader,model,loss_fn,optimizer):
    size=len(dataloader.dataset)
    model.train()
    for batch,(X,y) in enumerate(dataloader):
        X,y=X.to(device),y.to(device)
        #Compute prediction error
        pred=model(X)
        loss=loss_fn(pred,y)
        #Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch%100 == 0:
            loss,current=loss.item(),(batch+1)*len(X)
            print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")
#testing the model's performance against test dataset
def test(dataloader,model,loss_fn):
    size=len(dataloader.dataset)
    num_batches=len(dataloader)
    model.eval()
    test_loss,correct=0,0
    with torch.no.grad(): #w/o gradient computation
        for X,y in dataloader:
            X,y=X.to(device),y.to(device)
            pred=model(X)
            test_loss+=loss_fn(pred,y).item()
            correct+=(pred.argmax(1)==y).type(torch.float).sum().item() #pred.argmax(1) picks the class with highest score
    test_loss/=num_batches
    correct/=num_batches
    print(f"Test Error: \nAccuracy:{(100*correct):>0.1f}%, Avg loss:{test_loss:>8f}\n")
epochs=5
for t in range(epochs):
    print(f"Epoch{t+1}\n--------------------------------------")
    train(train_dataloader,model,loss_fn,optimizer)
    test(test_dataloader,model,loss_fn)
print("Done!")

torch.save(model.state_dict(),"model.pth")
print("Saved PyTorch model state to model.pth")

#Loading models
model=NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth",weights_only=True))

classes=[
    "T-shirt/Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
]

model.eval()
x,y=test_data[0][0], test_data[0][1]
with torch.no.grad():
    x=x.to(device)
    pred=model(x)
    predicted,actual=classes[pred[0].argmax(0)],classes[y]
    print(f"Predicted:{predicted},Actual:{actual}")
