#tensors
import numpy as np
import torch


#tensors initialization
#1)normal data
data=[[1.78,2.65],[3.89,4.09]]
x_data=torch.tensor(data) #converting them to tensor

#2)from numpy array
np_array=np.array(data)
x_np=torch.from_numpy(np_array)

#3)tensor from another tensor
#the new tensor retains the properties of the previous tensor, unless overriden explicitly.
x_ones=torch.ones_like(x_data) #retains the same property of x_data
print(f"Ones Tensor: {x_ones}\n")
# x_rand=torch.rand_like(x_data,dtype=torch.int) #mention dtype to override -doesn't work because the rand_like takes numbers
#within the uniform distribution of  [0.0,1.0)
x_rand=torch.rand_like(x_data,dtype=torch.float)
print(f"Random Tensor: {x_rand}\n")

#shape is a tuple of tensor dimensions- shapes determine the dimensions of the output tensor
shape=(2,3,)
rand_tensor=torch.rand(shape) #creates a tensor with shape mentioned above and fills it w random values
ones_tensor=torch.ones(shape)
zeros_tensor=torch.zeros(shape)
print(f"Random tensor:\n {rand_tensor}\n")
print(f"Ones tensor:\n {ones_tensor}\n")
print(f"Zeros tensor:\n {zeros_tensor}\n")

#3 main attributes of a tensor
tensor= torch.rand(3,4) #random nos randomly sampled from the interval 0,1
print(f"Shape of tensor:{tensor.shape}")
print(f"Datatype of tensor:{tensor.dtype}")
print(f"Device tensor is stored on:{tensor.device}")

#Operations on tensor
#by default the tensors are created on the CPU- e need to explicitly move the tensors to the accelerators using the .to method
#move the tensor to the current accelerator if available:
if torch.accelerator.is_available():
    tensor=torch.to(torch.accelerator.current_accelerator())

#Numpy -like indexing and slicing
tensor=torch.ones(4,4)
print(f"First row:{tensor[0]}")
print(f"First column:{tensor[:,0]}")
print(f"Last column:{tensor[...,-1]}") #... means ellipsis-selecting all leading dimensions,so it extracts the last element
                                        #in the last dimension (here it's the last column)
tensor[:,1]=0 #sets the second column to zero
print(tensor)

#Joining the tensors
t1=torch.cat([tensor,tensor,tensor],dim=1)
print(t1) #torch.cat concatenates a sequence of tensors along a given dimension

#ARITHMETIC OPS
#This computes the matrix multiplication btw two tensors - @
y1=tensor@tensor.T
y2=tensor.matmul(tensor.T)
y3=torch.rand_like(y1)
torch.matmul(tensor,tensor.T,out=y3) #this version stores the resultant of the matrix multiplication in y3

#element-wise operation(multiplication)
z1=tensor*tensor #multiplies w itself
z2=tensor.mul(tensor) # mul instead of *
z3=torch.rand_like(tensor)
torch.mul(tensor,tensor,out=z3) #stores result in z3

#single element tensors?- sometimes, we perform an operation on a tensor that reduces it to a single number
#(eg: summing all elements- the result will be still a tensor but with 1 element
agg=tensor.sum() #agg is a tensor with just one value inside
agg_item=agg.item() #if we want to convert this tensor into a regular number use .item()
print(agg_item,type(agg_item))

#inplace operations
#tensor.add_() - the results overwrites the original tensor other egs are copy_(),t_()
tensor.add_(5) #this adds 5 to every element in the tensor and modifies it directly
#avoid using inplace operations as much as possible, as in autograd, this might interfere and the orignal values might be lost
#in the computational graph, leading to wrong learning

#bridge with numpy- tensors on the cpu and the numpy arrays can share their underlying memory locations,changing one will change the otehr

#tensor to numpy array
t=torch.ones(5)
print(f"t:{t}")
n=t.numpy()
print(f"n:{n}")
#a change in the tensor array will reflect in the numpy array as well
t.add(1)
print(f"t:{t}")
print(f"n:{n}") #this wd be changed too

#Numpy array to tensor
n=np.ones(5)
t=torch.from_numpy(n) #converts a numpy array into a pytorch tensor which shares the same memory
#any change that we make in the numpy array will directly affect the tensor and vice versa
#no data is copied, it's a VIEW over the same underlying data
#eg:
np_array=np.array([1,2,3])
tensor=torch.from_numpy(np_array)

np_array[0]=100
print(f"tensor:{tensor}") #the tensor updates too
#this works only with the numpy arrays of the numeric types- eg: float32,int64
#if we want to break the link and clone the data- we can do .clone()


