#Implementation of convolution operation and max pooling operation from scratch (Assume a 3x3 kernel and apply it on an input image of 32x32.)
#
import numpy as np

def conv2d(image,kernel,stride=1,padding=1):
    h,w=image.shape
    fh,fw=kernel.shape

    #pad the image
    if padding >0:
        #syntax np.pad((array,((top,bottom),(left,right)),mode)
        #constant means it adds zeros
        padded_image=np.pad(image,((padding,padding),(padding,padding)),mode='constant')
    else:
        padded_image=image

    h_pad,w_pad=padded_image.shape
    #output dimensions
    out_h=((h_pad-fh)//stride)+1
    out_w=((w_pad-fw)//stride)+1
    output=np.zeros((out_h,out_w),dtype=float)

    #conv ops
    for i in range(0,out_h):
        for j in range(0,out_w):
            #extract the patch
            patch=padded_image[i*stride:i*stride+fh,j*stride:j*stride+fw]
    #elementwise multiplication
            output[i,j]=np.sum(patch*kernel)
    return output

def maxpool2d(image,pool=2,stride=2):
    h,w=image.shape
    out_h=(h-pool)//stride+1
    out_w=(w-pool)//stride+1
    output=np.zeros((out_h,out_w))
    for i in range(0,h-pool+1,stride):
        for j in range(0,w-pool+1,stride):
            window=image[i:i+pool,j:j+pool]
            output[i//stride,j//stride]=np.max(window)
    return output

np.random.seed(42)
image=np.random.randint(0,255,size=(32,32)) #input image
kernel=np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
conv_op=conv2d(image,kernel,stride=1,padding=1)
print(f"After convolution : {conv_op}")

pooling=maxpool2d(image,pool=2,stride=2)
print(f"After max pooling : {pooling}")

print(f"Conv output shape:{conv_op.shape}")
print(f"Max pooling output shape:{pooling.shape}")
