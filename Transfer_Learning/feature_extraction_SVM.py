#Exercise 3
#building a cifar classifier using pretrained resnet18 as feature extractor and feeding those to SVM to classify
import torch
from torchvision import datasets, transforms,models
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm

os.environ['http_proxy']="http://245hsbd015%40ibab.ac.in:Veda%402002@proxy.ibab.ac.in:3128/"
os.environ['https_proxy']="http://245hsbd015%40ibab.ac.in:Veda%402002@proxy.ibab.ac.in:3128/"

#device
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
print(model)
# or we can use weights=pretrained=True
model= model.to(device)

#freezing weights
model.eval()
for param in model.parameters():
    param.requires_grad=False #since we freeze (don't update the weights)

#extracting the feature
feature_extractor=torch.nn.Sequential(*list(model.children())[:-1]) #slicing- takes all layer upto last but 1

def extract_features(dataloader,desc="Extracting features"):
    features=[]
    labels=[]
    with torch.no_grad(): #no backprop since only FE
        for images,targets in tqdm(dataloader, desc=desc):
            images=images.to(device)
            feats=feature_extractor(images)
            # print(f"Shape after feature extraction: {feats.shape}") #this is ([62,512,1,1])
            feats=feats.view(feats.size(0),-1) #flattens the image from (batch_size,512,1,1) -> (batch_size,512)
            # print(f"Shape after flattening: {feats.shape}")
            features.append(feats.cpu().numpy()) #moves the features back to cpu, inorder to be trained for the SVM
            labels.append(targets.numpy()) #also convert them to numpy format
    features=np.concatenate(features,axis=0)
    labels=np.concatenate(labels,axis=0)
    return features,labels

#split the train and test dataset for the svms
X_train,y_train=extract_features(train_loader,desc="Train features")
X_test,y_test=extract_features(test_loader,desc="Test features")

print(f"Feature shape: {X_train.shape}") #this should be 50000,512
#predicting using one-vs-rest by default svc
print("Training SVM (OVR)...")
svm_clf=SVC(kernel='linear',decision_function_shape='ovr')
svm_clf.fit(X_train,y_train)
#prediction
y_pred=svm_clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"SVM Test Accuracy (OVR): {accuracy*100:.4f}%")
#using one-vs-one
print("Training SVM (OVR)...")
svm_clf = SVC(kernel='linear', decision_function_shape='ovo')
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Test Accuracy (OVO): {accuracy*100:.4f}%")

#optional plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 1️⃣ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix of LinearSVC on CIFAR-10 Features")
plt.show()

# 2️⃣ 2D Feature Visualization using PCA or TSNE
# Reduce features to 2D for plotting
pca = PCA(n_components=50)  # optional first step for faster t-SNE
X_train_pca = pca.fit_transform(X_test)

tsne = TSNE(n_components=2, random_state=42)
X_test_2D = tsne.fit_transform(X_train_pca)

plt.figure(figsize=(12,10))
scatter = plt.scatter(X_test_2D[:,0], X_test_2D[:,1], c=y_test, cmap='tab10', alpha=0.6)
plt.legend(handles=scatter.legend_elements()[0], labels=list(range(10)), title="Classes")
plt.title("t-SNE of ResNet18 Features for CIFAR-10 Test Set")
plt.show()
