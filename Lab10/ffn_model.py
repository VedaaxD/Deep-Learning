import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import dropout
from torch.utils.data import TensorDataset, DataLoader
import itertools
from torch.nn import functional as F


# #drop the repeated columns and the columns which are not required, in this dataset columns 0 and 1 are repeated.
# #data preprocessing
# #making the 1st column as index and transposing
# Loading and transposing
input = pd.read_csv("landmark_genes.csv",sep='\t', index_col=0)
output = pd.read_csv("target_genes.csv",sep='\t', index_col=0)
# Drop metadata cols (first 4)
input = input.iloc[:, 4:]  #keep only expression values
# Transpose so samples = rows, genes = columns
input = input.T
output = output.iloc[:, 4:]
output = output.T

input = input.apply(pd.to_numeric, errors="coerce")
output=output.apply(pd.to_numeric, errors="coerce")

# print("NaNs after conversion:", landmark_df.isna().sum().sum())
# Fill NaNs with column mean
input= input.fillna(input.mean())
output= output.fillna(output.mean())
print(input.nunique()) #shows unique value per column
#train,test and val split
X_train,X_tv,y_train,y_tv=train_test_split(input,output,test_size=0.2,random_state=42)
X_val,X_test,y_val,y_test=train_test_split(X_tv,y_tv,test_size=0.5,random_state=42)

# #data normalization
# scaler=StandardScaler()
# X_train_scaled=scaler.fit_transform(X_train)
# X_val_scaled=scaler.transform(X_val)
# X_test_scaled=scaler.transform(X_test)
#z score normalization
x_mean=X_train.mean()
x_std=X_train.std()

X_train=(X_train-x_mean)/x_std
X_val=(X_val-x_mean)/x_std
X_test=(X_test-x_mean)/x_std

y_mean=y_train.mean()
y_std=y_train.std()

y_train=(y_train-y_mean)/y_std
y_val=(y_val-y_mean)/y_std
y_test=(y_test-y_mean)/y_std

#converting these to tensor
X_train=torch.tensor(X_train.values,dtype=torch.float32)
X_val=torch.tensor(X_val.values,dtype=torch.float32)
X_test=torch.tensor(X_test.values,dtype=torch.float32)

y_train=torch.tensor(y_train.values,dtype=torch.float32)
y_test=torch.tensor(y_test.values,dtype=torch.float32)
y_val=torch.tensor(y_val.values,dtype=torch.float32)

print(f"Train:{X_train.shape},{y_train.shape}")
print(f"Val:{X_val.shape},{y_val.shape}")
print(f"Test:{X_test.shape},{y_test.shape}")

#Loading the dataset
train_data=TensorDataset(X_train,y_train)
test_data=TensorDataset(X_test,y_test)
val_data=TensorDataset(X_val,y_val)

#using dataloaders to create an iterable
train_loader=DataLoader(train_data,batch_size=64,shuffle=True) #adjust batch size and check the results to optimize
test_loader=DataLoader(test_data,batch_size=64,shuffle=True)
val_loader=DataLoader(val_data,batch_size=64,shuffle=True)

#Defining the model
class Model(nn.Module):
    def __init__(self,input_shape,output_shape,hidden_dim=32,dropout=0.2):
        super(Model, self).__init__()
        self.linear1=nn.Linear(input_shape,hidden_dim)
        self.bn1=nn.BatchNorm1d(hidden_dim)
        self.linear2=nn.Linear(hidden_dim,hidden_dim*2)
        self.bn2=nn.BatchNorm1d(hidden_dim*2)
        self.dropout=nn.Dropout(0.2)
        self.linear3=nn.Linear(hidden_dim*2,output_shape)

    def forward(self,x):
        x=self.linear1(x)
        x=self.bn1(x)
        x=F.relu(x)
        x=self.dropout(x)
        x=self.linear2(x)
        x=self.bn2(x)
        x=F.relu(x)
        x=self.dropout(x)
        x=self.linear3(x)
        return x
def train_model(model,train_loader,val_loader,lr=0.001,epochs=10):
    #defining optimizer and loss function
    optimizer=torch.optim.Adam(model.parameters(),lr=lr) #this is a version of GD which adapts learning rate per weight
    loss_func=nn.MSELoss()
    #training loop
    for epoch in range(epochs):
        model.train()
        for X,y in train_loader:
            optimizer.zero_grad() #this is essential to reset the old gradients to prevent accumulation
            pred=model(X) #fwd pass
            loss=loss_func(pred,y)
            loss.backward() #backprop
            optimizer.step() #this uses the calculated gradients and the specific update rule defined by the optimizer
                            #to adjust the values of the model's parameters -weights and biases..inorder to minimize the loss function

    #validation loop
    model.eval() #no dropouts, no batchnorm
    val_loss=0.0
    with torch.no_grad(): #no gradient tracking for the evaluation
        for X,y in val_loader:
            pred=model(X)
            loss=loss_func(pred,y) #calculates avg loss across all samples in each batch
            val_loss+=loss.item()*X.size(0) #loss.item() extracts the numerical value from the tensor and
            #detaches the value from the computation graph - now it's just python float
            #X.size(0) gets the number of samples in the current batch
            #val_loss - reverses the average giving the total loss sum for the batch for evaluation(as we don't update weights in val/test)
    val_loss=val_loss/len(val_loader.dataset) #in sklearn, model.accuracy/score does this step inetrnally and gives the final loss
    return val_loss

def hyperparameter_tuning(X_train,y_train,X_val,y_val,input_dim,output_dim):
    param_grid={
        "hidden_dim":[32,64,128],
        "dropout":[0.2,0.3,0.5],
        "lr":[0.0001,0.001,0.01],
        "batch_size":[32,64,128]
    }
    best_val_loss=float("inf")
    best_params=None
    best_state=None
    #itertools.product tries out all combinations
    for hidden_dim,dropout,lr,batch_size in itertools.product(
        param_grid["hidden_dim"],
        param_grid["dropout"],
        param_grid["lr"],
        param_grid["batch_size"]
    ):
        print(f"Training with hidden_dim:{hidden_dim},lr:{lr},batch_size:{batch_size}")

        train_data=TensorDataset(X_train,y_train)
        val_data=TensorDataset(X_val,y_val)

        train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
        val_loader=DataLoader(val_data,batch_size=batch_size,shuffle=True)

        model=Model(input_dim,output_dim,hidden_dim=hidden_dim,dropout=dropout)
        val_loss=train_model(model,train_loader,val_loader,lr=lr,epochs=10)
        print(f"Validation loss:{val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss=val_loss
            best_params=(hidden_dim,dropout,lr,batch_size)
            best_state=model.state_dict()

    print(f"\n Best hyperparameters:{best_params} with validation loss:{best_val_loss:.4f}")
    return best_params,best_state

input_dim=X_train.shape[1]
output_dim=y_train.shape[1]
print(output_dim)

#Best hyperparameters
best_params,best_state=hyperparameter_tuning(X_train,y_train,X_val,y_val,input_dim,output_dim)
hidden_dim,dropout,lr,batch_size=best_params
#combine train+val to retrain on the train set (train+val=train)
train_and_val_data=TensorDataset(torch.cat([X_train,X_val]),torch.cat([y_train,y_val]))
#torch.cat - concatenates the train and val dataset
train_and_val_loader=DataLoader(train_and_val_data,batch_size=batch_size,shuffle=True,drop_last=True)
#drop_last=True - if the last batch is smaller than the batch size, we can drop it - which helps the batchnorm stability.
test_loader=DataLoader(TensorDataset(X_test,y_test),batch_size=batch_size)

#final model
final_model=Model(input_dim,output_dim,hidden_dim=hidden_dim,) #change dropout
optimizer=torch.optim.Adam(final_model.parameters(),lr=lr)
loss_fn=nn.MSELoss()

#Retraining the model on train and val set with best parameters obt from val set
for epoch in range(10):
    final_model.train()
    train_loss=0.0
    for X,y in train_and_val_loader:
        optimizer.zero_grad()
        pred=final_model(X)
        loss=loss_fn(pred,y)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()*X.size(0)
    train_loss=train_loss/len(train_and_val_loader.dataset)
    #on each epoch  how much will be the training loss (to visualize if model converges properly)
    print(f"Epoch{epoch+1}:Train loss{train_loss:.4f}")

#test evaluation
final_model.eval()
test_loss=0.0
with torch.no_grad():
    for X,y in test_loader:
        pred=final_model(X)
        loss=loss_fn(pred,y)
        test_loss+=loss.item()*X.size(0)
test_loss=test_loss/len(test_loader.dataset)

print(f"Test loss:{test_loss:.4f}")

print(X_train[:5])
print(X_test[:5])