# Implement CNN using PyTorch for image classification using cifar10 dataset - https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html .
# Plot train error vs increasing number of layers. After some point, the training error increases with the number of layers.
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
os.environ['http_proxy']="http://245hsbd015%40ibab.ac.in:Veda%402002@proxy.ibab.ac.in:3128/"
os.environ['https_proxy']="https://245hsbd015%40ibab.ac.in:Veda%402002@proxy.ibab.ac.in:3128/"

#Data preparation- CIFAR10
#Transform: convert images to Tensor and normalize
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#mean and stds for 3 channels R,G and B
batch_size=4 #4 images per training step
trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=2)
#num workers mean the no of subprocesses for loading the data
# for images,labels in trainloader: trainloader is an iterator, we can loop over it
#     images.shape=[batch_size,3,32,32]
#     labels.shape=[batch_size]
testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testloader=torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=2)

#class labels in cifar10
classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

#helper functions to show the images
def imshow(img):
    img=img/2 +0.5 #unnormalize back to [0,1]
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0))) #convert the format from C,H,W to H,W,C
    plt.show()

#Get a batch of random training images and show them
dataiter=iter(trainloader)
images,labels=next(dataiter)
#showing the images
imshow(torchvision.utils.make_grid(images))
#print the labels
print(''.join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))

#Defining a CNN model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.fc1=nn.Linear(16*5*5,120) #16 feature maps of 5X5 each
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=torch.flatten(x,1) #flatten all dimensions except batch
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x) #outputs-logits
        return x
net=Net()
#define loss function
criterion=nn.CrossEntropyLoss() #multi-class classification loss
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9) #Stochastic gradient descent

#training the network
for epoch in range(2): #looping over the dataset multiple times
    running_loss=0.0
    for i,data in enumerate(trainloader,0):
        #Get inputs -images and labels
        inputs,labels=data
        #zero gradients
        optimizer.zero_grad()
        #forward pass
        outputs=net(inputs)
        #compute loss
        loss=criterion(outputs,labels)
        #backprop
        loss.backward()
        #update parameters
        optimizer.step()
        running_loss += loss.item()
        if i%2000==1999:#print loss every 2000 mini-batches
            print(f"[{epoch+1},{i+1:5d}] loss:{running_loss/2000:.3f}")
            running_loss=0.0
print("Finished Training")
#Save the trained model
PATH="./cifar_net.pth" #the learned parameters will be stored in the .pth file, we can reuse it to directly run the test set.
#saves time and computation.
torch.save(net.state_dict(), PATH)

#testing the network on the test data
dataiter=iter(testloader)
images,labels=next(dataiter)
#print the images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth:',''.join(f'{classes[labels[j]]:5s}' for j in range(4)))

net=Net()
net.load_state_dict(torch.load(PATH)) #load trained weights
outputs=net(images)

_,predicted=torch.max(outputs,1)
print('Predicted:',''.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

#Test accuracy on the full dataset
#we need to see how the network performs in whole dataset
correct=0
total=0
#since we're not training we don't need tp calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images,labels=data
        #calculate outputs by running images through the network
        outputs=net(images)
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
print(f'Accuracy of the network on the 10000 test images:{100*correct//total}%')
#we can evaluate which classes have performed well and which did not
#count predictions for each class
correct_pred={classname:0 for classname in classes}
total_pred={classname:0 for classname in classes}
#no gradients required for the test
with torch.no_grad():
    for data in testloader:
        images,labels=data
        outputs=net(images)
        _,predictions=torch.max(outputs,1)
        #collect the correct predictions for each class
        for label,prediction in zip(labels,predictions):
            if label==prediction:
                correct_pred[classes[label]]+=1
            total_pred[classes[label]]+=1
#print accuracy for each class
for classname,correct_count in correct_pred.items():
    accuracy=100*float(correct_count)/total_pred[classname]
    print(f'Accuracy for class {classname:5s} is {accuracy:.1f}%')

