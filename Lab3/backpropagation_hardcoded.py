#Backpropagation-hardcoded
import numpy as np
def inputs():
    x=-2
    y=5
    z=-4
    return x,y,z

def arithmetic_ops(x,y,z):
    f1=x+y
    f=f1*z
    return f,f1

def local_gradients(x,y,z):
    f1=x+y
    f=(f1)*z #function
    df_df1=z
    df_dz=f1
    df_dy=1*z #local*global
    df_dx=1*z #local*global
    return df_dx,df_dy,df_dz,df_df1

def backprop(df_dx,df_dy,df_dz,df_df1):
    print(f"Gradient: df/df1={df_df1}")
    print(f"Gradient: df/dz={df_dz}")
    print(f"Gradient: df/dx={df_dx}")
    print(f"Gradient: df/dy={df_dy}")

def main():
    x,y,z=inputs()
    f,f1=arithmetic_ops(x,y,z)
    df_dx,df_dy,df_dz,df_df1=local_gradients(x,y,z)
    backprop(df_dx,df_dy,df_dz,df_df1)

if __name__ == "__main__":
    main()



