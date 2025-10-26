#Image captioning using CNNs and RNNs (pytorch) for the flickr 8k dataset

#Step1: Retrieve the embedding vector for each image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
img_path="Images"
image_embedding_file="image_embedding.pt"
#preprocessing the image
transform=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])
#using the pretrained resnet-18
model=models.resnet18(pretrained=True) #depreceation warning
# model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT) #gives the most up-to-date weights.
#removing the classification layer(head)
model.fc=nn.Identity() #last layer remains unchanged [B,C] -(1,512)
model.eval()

def extract_features(img_path):
    img=Image.open(img_path).convert('RGB')
    img=transform(img).unsqueeze(0) #this will add batch dimension as resnet expects it -got an error (B,C,H,W) - shape will be (1,512,1,1)
    with torch.no_grad():
        embedding=model(img) #shape-[B,C] -(1,512)
    embedding=embedding.squeeze() #squeeze -removes the batch dimension as we are passing the images one by one
    return embedding
#loop over the images and save the embeddings
embeddings={}
for filename in os.listdir(img_path):
    if filename.lower().endswith((".jpg","jpeg","png")):
        path=os.path.join(img_path,filename)
        embeddings[filename]=extract_features(path)

#save it in the embedding file
torch.save(embeddings,image_embedding_file)
print(f"Saved {len(embeddings)} embeddings to {image_embedding_file}.")








