#Exercise 4
#building a cifar classifier using pretrained resnet18 but finetuning the model, where all layers are frozen (weights aren't updated)
#except the last few.
import torch
import torch.optim as optim
from torchvision import datasets, transforms,models
import numpy as np
from torch.utils.data import DataLoader
import os
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['http_proxy']="http://245hsbd015%40ibab.ac.in:Veda%402002@proxy.ibab.ac.in:3128/"
os.environ['https_proxy']="http://245hsbd015%40ibab.ac.in:Veda%402002@proxy.ibab.ac.in:3128/"

#image preprocessing
preprocess=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])
#dataset
train_data=datasets.CIFAR10(root='./data',download=True, train=True,transform=preprocess)
test_data=datasets.CIFAR10(root='./data',download=True, train=False,transform=preprocess)
print(f"loading the data..")
train_loader=DataLoader(train_data,batch_size=64,shuffle=True,num_workers=4)
test_loader=DataLoader(test_data,batch_size=64,shuffle=False,num_workers=4)

#loading the pretrained ResNet18
print(f"loading the pretrained ResNet model..")
model=models.resnet18(weights='IMAGENET1K_V1')
# or we can use weights=pretrained=True
model= model.to(device)

#we are freezing all the layers first
for param in model.parameters():
    param.requires_grad =False
#
#unfreeze only the last few layers
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad =True #as we can see from the print(model) the layer 4 is the last block before pooling and fc
        #so we are letting only the higher level features to be adapted to cifar-10

num_features=model.fc.in_features
model.fc =nn.Linear(num_features,10).to(device)

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=0.001) #filters out the parameters which aren't
#partcipating in the training (keeps only the parameters which are going to be trained (unfreezed ones)) and pass it on to adam optimizer

#Training loop
num_epochs=5
for epoch in range(num_epochs):
    model.train()
    running_loss=0.0
    # tqdm progress bar for train_loader because it is slow..
    loop=tqdm(train_loader,desc=f"Epoch[{epoch}/{num_epochs}]",leave=False)
    for images,labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()*images.size(0) #loss.item() - avg loss for a single batch
        #running loss means sum of batch losses so far

        # update tqdm bar with batch loss
        loop.set_postfix(batch_loss=loss.item()) #.set_postfix is a method that prints the text after the progress bar..
    # average loss across batches for one epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_loss:.4f}")

#testing loop
model.eval()
total=0
correct=0
with torch.no_grad():
    for images,labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

print(f"Test accuracy: {100*correct/total:.4f}")
