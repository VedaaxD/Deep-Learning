#Following the pytorch tutorial
# Datasets and dataloaders

#Dataset handles loading individual samples
#Dataloader- efficiently feeds the batches of data into your model, with options like shuffling and parallel processing

#Loading PRE-DEFINED DATASETS- FAashionMNIST
# import torch
# from torch.utils import Dataset
# from torchvision import datasets
# from torchvision import ToTensor
# import matplotlib.pyplot as plt
#
# training_data=datasets.FashionMNIST(
#     root="data", #where to store the dataset
#     train=True,
#     download=True,
#     transform=ToTensor() #converts the image dataset into PyTorch tensor
# )
# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor()
# )
# #Visualising the dataset
# #this part shows 9 random images from the training dataset with their labels like "Sneaker", "Bags"
# #our goal is to show some 3x3 grid of random images from the training_data dataset- with their label names as titles
# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3 #for total of 9 images
# #we can index the datasets manually like a list
# for i in range(1,cols*rows+1):
#     sample_idx=torch.randint(len(training_data),size=(1,)).item() #each time it picks a random index
#     img,label=training_data[sample_idx] #loading the corresponding image and its label from the training data
#     figure.add_subplot(cols,rows,i) #adding the subplot for each image
#     plt.title(labels_map[label])
#     plt.axis('off')
#     plt.imshow(img.squeeze(),cmap='gray')
# plt.show()

#Creating a Custom Dataset for your files
#this can be used when working with our own image files and not built-in datasets

#A custom dataset class must implement 3 functions- __init__, __len__ and __getitem__
#the FashionMNIST images are stored in a directory called img_dir and their labels are stored separately in CSV file-annotations file

import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self,annotations_file,img_dir,transform=None,target_transform=None):
        self.img_labels=pd.read_csv(annotations_file) #pandas for easy indexing
        self.img_dir=img_dir
        self.transform=transform #transformation for input images- like resize and normalize
        self.target_transform=target_transform #transformation for the labels- one hot encoding

    def __len__(self): #retutns the total number of samples in the dataset
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path=os.path.join(self.img_dir,self.img_labels.iloc[idx,0])
        image=decode_image(img_path)
        label=self.img_labels.iloc[idx,1]
        if self.transform: #if transform is provided it applies to the image
            image=self.transform(image)
        if self.target_transform: #similarly for the target
            label=self.target_transform(label)
        return image, label
#Paths to csv and images
train_annotations="train_labels.csv"
test_annotations="test_labels.csv"
train_img_dir="train"
test_img_dir="test"
#Create Dataset objects
training_data=CustomImageDataset(train_annotations,train_img_dir)
test_data=CustomImageDataset(test_annotations,test_img_dir)
#Create dataloaders
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")