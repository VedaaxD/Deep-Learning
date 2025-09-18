#Implementation of RNN from scratch - flexibility to handle user inputs
import numpy as np

def rnn_forward_pass(W_hh,W_xh,W_hy,X,h0):
    timesteps=X.shape[0]
    hidden_dim=W_hh.shape[0]
    output_dim=W_hy.shape[0]
    h=np.zeros((timesteps,hidden_dim))
    y=np.zeros((timesteps,output_dim))
    h_prev=h0 #initializing

    for t in range(timesteps):
        x_t=X[t]
        h[t]=np.tanh(np.dot(W_hh,h_prev)+np.dot(W_xh,x_t))
        h_t=h[t]
        y[t]=np.dot(W_hy,h_t)
        y_t=y[t]
        h_prev=h_t #for the next iteration
    return h,y

def main():
    n_samples=int(input("Enter the number of samples: "))#X1 (word) can contain x1,x2,x3 etc and X2(word/sample) can contain x1,x2,x3
    timesteps=int(input("Enter the number of timesteps: "))#no of timesteps x1,x2,x3,x4 etc
    input_dim=int(input("Enter the input dimension which is fixed across timesteps: ")) #x1=2x1 , x2=2x1 fixed
    n_hidden=int(input("Enter the number of hidden units: ")) #h-1,2,3 etc
    output_dim=int(input("Enter the output dimension: ")) #y1=2x1

    #initializing the weights
    W_hh=np.random.randn(n_hidden,n_hidden) #3x3 like in class example
    print("W_hh=",W_hh)
    W_xh=np.random.randn(n_hidden,input_dim) #3x2
    print("W_xh=",W_xh)
    W_hy=np.random.randn(output_dim,n_hidden) #2x3
    print("W_hy=",W_hy)
    for sample_index in range(n_samples):
        #generate a representation vector for each sample- which shd be the same across all the time steps
        print(f"Sample:\n{sample_index+1}")
        choice=int(input("Enter '1' to manually enter inputs or '2' to generate inputs randomly: "))
        if choice==1:
            X=np.zeros((timesteps,input_dim))
            print(f"Enter {timesteps} input of dimension {input_dim}:")
            for t in range(timesteps):
                while True:
                    row_input=input(f"Timestep {t+1}: Enter space-separated values of dimension {input_dim}: ")
                    values=row_input.strip().split()

                    if len(values)==input_dim: #checking the dimensional errors from the user
                        try:
                            X[t]=[float(v) for v in values] #checking the value errors from the user
                            break
                        except ValueError as v:
                            print(v)
                            print(f"Please enter only numeric values.")
                    else:
                        print("Invalid dimension/format.")
        else:
            X=np.random.randn(timesteps,input_dim) # or X=np.random.randn(timesteps,1)
            print(f"Randomly generated inputs:\n{X}")

        #initial hidden state h0
        h0=np.ones(n_hidden)

        h,y=rnn_forward_pass(W_hh,W_xh,W_hy,X,h0)
        for h_state in h:
            print(f"Hidden states:{h}")

        #for each time step t,y

        print("Outputs:")
        for t in range(timesteps):
            print(f"At timestep {t+1}:{y[t]}")
if __name__=="__main__":
    main()
