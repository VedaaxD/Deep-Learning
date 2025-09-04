# Download MNIST dataset and implement a MNIST classifier using CNN PyTorch library.
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm
import os

os.environ['http_proxy']="http://245hsbd015%40ibab.ac.in:Veda%402002@proxy.ibab.ac.in:3128/"
os.environ['https_proxy']="https://245hsbd015%40ibab.ac.in:Veda%402002@proxy.ibab.ac.in:3128/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#image preprocessing
preprocess=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5))
])
#Loading the data
print(f"Loading the MNIST dataset...")
train_data=datasets.MNIST(root='./data',train=True,transform=preprocess,download=True)
test_data=datasets.MNIST(root='./data',train=False,transform=preprocess,download=True)
#trainloader and testloader
train_loader=torch.utils.data.DataLoader(dataset=train_data,batch_size=64,shuffle=True,num_workers=4)
test_loader=torch.utils.data.DataLoader(dataset=test_data,batch_size=64,shuffle=False,num_workers=4)

class MNIST_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_Classifier,self).__init__()
        #First conv layer
        self.conv1=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=7,stride=1,padding=3)
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool2d(3,2)
        #Second conv layer
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=7,stride=1,padding=3)
        self.relu2=nn.ReLU()
        self.pool2=nn.MaxPool2d(3,2)
        #Third conv layer
        self.conv3=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1)
        self.relu3=nn.ReLU()
        self.pool3=nn.MaxPool2d(3,2)
        self.fc1=nn.Linear(in_features=32,out_features=64)
        self.relu4=nn.ReLU()
        self.fc2=nn.Linear(in_features=64,out_features=10)


    def forward(self,x):
        x=self.pool1(self.relu1(self.conv1(x)))
        x=self.pool2(self.relu2(self.conv2(x)))
        x=self.pool3(self.relu3(self.conv3(x)))
        x=x.view(x.size(0),-1) #flatten out for the further fc layers
        x=self.relu4(self.fc1(x))
        x=self.fc2(x)
        return x

#instantiating the model
model=MNIST_Classifier()
#define the loss function and the optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

#train loop
num_epochs=5
for epoch in range(num_epochs):
    running_loss=0.0
    loop=tqdm(train_loader,desc=f"epoch [{epoch}/{num_epochs}]",leave=False)
    for images,labels in loop:
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs=model(images)
        loss=criterion(outputs, labels)
        #Backward pass and optimization
        loss.backward()
        optimizer.step()
        #Accumulate and display the loss
        running_loss +=loss.item()*images.size(0)
        loop.set_postfix(loss=loss.item())
    #avg loss for a epoch
    avg_epoch_loss=running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_epoch_loss:.4f}")
print("Training complete!")

print("\nEvaluating the model...")
#testing loop
model.eval()
correct=0
total=0
with torch.no_grad():
    for images,labels in test_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs,1) #don't use outputs.data
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
print(f"Test Accuracy:{100*correct/total:.4f}")


