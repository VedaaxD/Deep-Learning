#Exercise 1
#building a cifar classifier using pretrained resnet18 as feature extractor and modifying the last layer as FC layer.
import torch
from torch import nn ,optim
from torchvision import datasets,transforms,models
import os
os.environ['http_proxy']="http://245hsbd015%40ibab.ac.in:Veda%402002@proxy.ibab.ac.in:3128/"
os.environ['https_proxy']="http://245hsbd015%40ibab.ac.in:Veda%402002@proxy.ibab.ac.in:3128/"

#image preprocessing
preprocess=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])
#loading the data
train_data=datasets.CIFAR10(root='./data',download=True, train=True,transform=preprocess)
test_data=datasets.CIFAR10(root='./data',download=True, train=False,transform=preprocess)
#
print(f"loading the data..")
#trainloader and testloader
train_loader=torch.utils.data.DataLoader(dataset=train_data,batch_size=64,shuffle=True,num_workers=4)
test_loader=torch.utils.data.DataLoader(dataset=test_data,batch_size=64,shuffle=False,num_workers=4)
print(f"loading the pretrained model..")
#loading the pretrained dataset
model=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
for param in model.parameters():
    param.requires_grad=False #because it is a feature extractor

#to get the input features of the last layer
num_features=model.fc.in_features

#just replace the final FC layer
model.fc=nn.Linear(num_features,10) #as cifar has 10 classes

#only train the new fc layer
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.fc.parameters(),lr=0.001)
print(f"Training the model..")
#Training loop
# num_epochs=5
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0  # accumulate loss for each batch
#     for images,labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs=model(images)
#         loss=criterion(outputs,labels)
#         loss.backward()
#         optimizer.step()
#     running_loss += loss.item() * images.size(0)  # accumulate total loss
#     # average loss for the epoch
#     epoch_loss = running_loss / len(train_loader.dataset)
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")
from tqdm import tqdm
# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # tqdm progress bar for train_loader because it is slow..
    loop=tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
    for images,labels in loop:
        images,labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        # update tqdm bar with batch loss
        loop.set_postfix(batch_loss=loss.item())
    # average loss for the epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_loss:.4f}")

print(f"Evaluating the model..")
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
