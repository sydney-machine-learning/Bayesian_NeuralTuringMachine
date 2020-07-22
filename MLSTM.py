import torch
from prettytable import PrettyTable
import time

class MFNN2(torch.nn.Module):
    
    def __init__(self, P_input_dims, Q_input_dims, output_dims):
        super().__init__()
        
        #assert len(P_input_dims) == len(output_dims) == len(Q_input_dims) == 2
        
        self.P_input_dims = P_input_dims # [r1, c1]
        self.Q_input_dims = Q_input_dims # [r2, c2]
        self.output_dims = output_dims # [r_prime, c_prime]

        
        #Parameters of the model
        self.Up = torch.nn.Parameter( torch.empty([output_dims[0], P_input_dims[0]])  ) #Shape = [r_prime, r1]
        torch.nn.init.kaiming_uniform_(self.Up)
        
        self.Vp = torch.nn.Parameter( torch.empty([P_input_dims[1], output_dims[1]])  ) #Shape = [c1, c_prime]
        torch.nn.init.kaiming_uniform_(self.Vp)

        self.Uq = torch.nn.Parameter( torch.empty([output_dims[0], Q_input_dims[0]])  ) #Shape = [r_prime, r2]
        torch.nn.init.kaiming_uniform_(self.Uq)
        
        self.Vq = torch.nn.Parameter( torch.empty([Q_input_dims[1], output_dims[1]])  ) #Shape = [c2, c_prime]
        torch.nn.init.kaiming_uniform_(self.Vq)
        
        self.B = torch.nn.Parameter( torch.empty(output_dims)  ) #Shape = [r_prime, c_prime]
        torch.nn.init.kaiming_uniform_(self.B)
        
        
    def forward(self, P, Q):
        
        Y = torch.matmul( torch.matmul(self.Up, P), self.Vp ) + torch.matmul( torch.matmul(self.Uq, Q), self.Vq ) + self.B
        
        
        return Y


class MFNN3(torch.nn.Module):
    
    def __init__(self, P_input_dims, Q_input_dims, R_input_dims, output_dims):
        super().__init__()
        
        #assert len(P_input_dims) == len(output_dims) == len(Q_input_dims) == 2
        
        self.P_input_dims = P_input_dims # [r1, c1]
        self.Q_input_dims = Q_input_dims # [r2, c2]
        self.output_dims = output_dims # [r_prime, c_prime]

        
        #Parameters of the model
        self.Up = torch.nn.Parameter( torch.empty([output_dims[0], P_input_dims[0]])  ) #Shape = [r_prime, r1]
        torch.nn.init.kaiming_uniform_(self.Up)
        
        self.Vp = torch.nn.Parameter( torch.empty([P_input_dims[1], output_dims[1]])  ) #Shape = [c1, c_prime]
        torch.nn.init.kaiming_uniform_(self.Vp)

        self.Uq = torch.nn.Parameter( torch.empty([output_dims[0], Q_input_dims[0]])  ) #Shape = [r_prime, r2]
        torch.nn.init.kaiming_uniform_(self.Uq)
        
        self.Vq = torch.nn.Parameter( torch.empty([Q_input_dims[1], output_dims[1]])  ) #Shape = [c2, c_prime]
        torch.nn.init.kaiming_uniform_(self.Vq)

        self.Ur = torch.nn.Parameter( torch.empty([output_dims[0], R_input_dims[0]])  ) #Shape = [r_prime, r1]
        torch.nn.init.kaiming_uniform_(self.Ur)
        
        self.Vr = torch.nn.Parameter( torch.empty([R_input_dims[1], output_dims[1]])  ) #Shape = [c1, c_prime]
        torch.nn.init.kaiming_uniform_(self.Vr)

        
        self.B = torch.nn.Parameter( torch.empty(output_dims)  ) #Shape = [r_prime, c_prime]
        torch.nn.init.kaiming_uniform_(self.B)
        
        
    def forward(self, P, Q, R):
        
        Y = torch.matmul( torch.matmul(self.Up, P), self.Vp ) + torch.matmul( torch.matmul(self.Uq, Q), self.Vq ) + torch.matmul( torch.matmul(self.Ur, R), self.Vr ) + self.B
        
        
        return Y



        
        

class MLSTMCell2(torch.nn.Module):
    
    def __init__(self, input_dims, inner_dims, hidden_dims):
        """
        input_dims: The dimensions of the input, excluding batch ofc.
        
        inner_dims: This is equal to the Q_input_dims for MFNN2. It exists because in the case when one might want to concatenate two hidden matrices as done
                    in layered LSTM. Usually, inner_dims == hidden_dims, unless some special cases.
                    
        hidden_dims: The dimensions of the hidden state matrix.
        """
        
        
        super().__init__()
        
        self.input_dims = input_dims
        self.inner_dims = inner_dims
        self.hidden_dims = hidden_dims
        
        self.Weights = torch.nn.ModuleDict({
            
            'input_t' : MFNN2(P_input_dims=self.input_dims, Q_input_dims=self.inner_dims, output_dims=self.hidden_dims),
            'forget_t' : MFNN2(P_input_dims=self.input_dims, Q_input_dims=self.inner_dims, output_dims=self.hidden_dims),
            'output_t' : MFNN2(P_input_dims=self.input_dims, Q_input_dims=self.inner_dims, output_dims=self.hidden_dims),
            'cell_t' : MFNN2(P_input_dims=self.input_dims, Q_input_dims=self.inner_dims, output_dims=self.hidden_dims),
            
        })
        
        
        
    def forward(self, X_t, H_prev, C_prev):
        """
        X_t : Tensor of shape [batch_size, *input_dims].
        H_prev : Hidden matrix at t-1, shape [batch_size, *hidden_dims] OR [batch_size, *inner_dims] when hidden_dims != inner_dims (At which point, it's not logical to call it H_prev, rather calling it Inner_prev is suitable.)
        C_prev : Hidden matrix at t-1, shape [batch_size, *hidden_dims].
        """
        
#         assert list(X_t.shape)[1:] == self.input_dims, 'Wrong Input shape {}!={}'.format(list(X_t.shape)[1:], self.input_dims)
#         assert list(C_prev.shape)[1:] == self.hidden_dims, 'Wrong shape for previous Cell State.'
#         assert list(H_prev.shape)[1:] == self.inner_dims, 'Wrong shape for previous Hidden State.'
        
        I_t = torch.sigmoid(self.Weights['input_t']( X_t, H_prev ))
        F_t = torch.sigmoid(self.Weights['forget_t']( X_t, H_prev ))
        O_t = torch.sigmoid(self.Weights['output_t']( X_t, H_prev ))
        C_hat_t = torch.sigmoid(self.Weights['cell_t']( X_t, H_prev ))
        
        C_t = F_t * C_prev + I_t * C_hat_t #Shape : [batch_size, *hidden_dims]
        
        H_t = O_t * torch.tanh(C_t) #Shape : [batch_size, *hidden_dims]
        
        return H_t, C_t 


class MLSTM2(torch.nn.Module):    # Layered MLSTM
    
    def __init__(self, input_dims, hidden_dims, num_layers, device=None):
        
        super().__init__()
        
        
        
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        
        self.Layers = torch.nn.ModuleList([ MLSTMCell2(input_dims=self.input_dims, inner_dims=[self.hidden_dims[0], 2*self.hidden_dims[1]], hidden_dims=self.hidden_dims) for _ in range(self.num_layers)])
                                                                                                                #  ^ as inner_dims is prev layer's H and this layer's H concatenated. 
            
        self.device = device
        
        
    def forward(self, X_t, H_prev, C_prev):
        
        """
        X_t : input matrix of shape [batch_size, *input_dims], this is timestep t's input matrix.
        H_prev : list of length self.num_layers, each instance is a tensor of shape [batch_size, *hidden_dims]. At t=0, make sure it is just a list of 0 matrix.
        C_prev : list of length self.num_layers, each instance is a tensor of shape [batch_size, *hidden_dims]. At t=0, make sure it is just a list of 0 matrix.
        
        RETURNS: 
                H_t_list : list of length self.num_layers, each instance is a tensor of shape [batch_size, *hidden_dims]
                C_t_list : list of length self.num_layers, each instance is a tensor of shape [batch_size, *hidden_dims]
        """
#         t1 = time.time()
    
        H_0_t = torch.zeros([X_t.shape[0], *self.hidden_dims], device=self.device) # For layer 0 at all time.
        
        appended_H_prev_list = [H_0_t, *H_prev]
        
        C_t_list = []
        
        for i in range(1, self.num_layers+1):
                                    #            Prev time, current layer ; Current time, prev layer 
                layer_Hidden_input = torch.cat( [ appended_H_prev_list[i], appended_H_prev_list[i-1] ] , dim = 2)
                
                H_t, C_t = self.Layers[i-1]( X_t, layer_Hidden_input, C_prev[i-1] )
            
                appended_H_prev_list[i] = H_t
                C_t_list.append(C_t)

#         t2 = time.time()
#         print("MLSTM called. Time taken: {}".format(t2-t1))
        
        
        return appended_H_prev_list[1:], C_t_list
        
        


class MLSTMCell3(torch.nn.Module):
    
    def __init__(self, input_dims, inner_dims_Q, inner_dims_R, hidden_dims):
        """
        input_dims: The dimensions of the input, excluding batch ofc.
        
        inner_dims: This is equal to the Q_input_dims for MFNN2. It exists because in the case when one might want to concatenate two hidden matrices as done
                    in layered LSTM. Usually, inner_dims == hidden_dims, unless some special cases.
                    
        hidden_dims: The dimensions of the hidden state matrix.
        """
        
        
        super().__init__()
        
        self.input_dims = input_dims
        self.inner_dims_Q = inner_dims_Q
        self.inner_dims_R = inner_dims_R
        self.hidden_dims = hidden_dims
        
        self.Weights = torch.nn.ModuleDict({
            
            'input_t' : MFNN3(P_input_dims=self.input_dims, Q_input_dims=self.inner_dims_Q, R_input_dims=self.inner_dims_R, output_dims=self.hidden_dims),
            'forget_t' : MFNN3(P_input_dims=self.input_dims, Q_input_dims=self.inner_dims_Q, R_input_dims=self.inner_dims_R, output_dims=self.hidden_dims),
            'output_t' : MFNN3(P_input_dims=self.input_dims, Q_input_dims=self.inner_dims_Q, R_input_dims=self.inner_dims_R, output_dims=self.hidden_dims),
            'cell_t' : MFNN3(P_input_dims=self.input_dims, Q_input_dims=self.inner_dims_Q, R_input_dims=self.inner_dims_R, output_dims=self.hidden_dims),
            
        })
        
        
        
    def forward(self, X_t, H_prev, ExtraInputMat, C_prev):
        """
        X_t : Tensor of shape [batch_size, *input_dims].
        H_prev : Hidden matrix at t-1, shape [batch_size, *hidden_dims] OR [batch_size, *inner_dims_Q] when hidden_dims != inner_dims_Q (At which point, it's not logical to call it H_prev, rather calling it Inner_prev is suitable.)
        C_prev : Hidden matrix at t-1, shape [batch_size, *hidden_dims].
        ExtraInputMat : Extra thrird Matrix to process of shape [batch_size, *inner_dims_R]
        """
        
#         assert list(X_t.shape)[1:] == self.input_dims, 'Wrong Input shape {}!={}'.format(list(X_t.shape)[1:], self.input_dims)
#         assert list(C_prev.shape)[1:] == self.hidden_dims, 'Wrong shape for previous Cell State.'
#         assert list(H_prev.shape)[1:] == self.inner_dims, 'Wrong shape for previous Hidden State.'
        
        I_t = torch.sigmoid(self.Weights['input_t']( X_t, H_prev, ExtraInputMat ))
        F_t = torch.sigmoid(self.Weights['forget_t']( X_t, H_prev, ExtraInputMat ))
        O_t = torch.sigmoid(self.Weights['output_t']( X_t, H_prev, ExtraInputMat ))
        C_hat_t = torch.sigmoid(self.Weights['cell_t']( X_t, H_prev, ExtraInputMat ))
        
        C_t = F_t * C_prev + I_t * C_hat_t #Shape : [batch_size, *hidden_dims]
        
        H_t = O_t * torch.tanh(C_t) #Shape : [batch_size, *hidden_dims]
        
        return H_t, C_t 
        
  
class MLSTM3(torch.nn.Module):    # Layered MLSTM

    def __init__(self, input_dims, hidden_dims, ExtraInput_dims, num_layers, device=None):
        
        super().__init__()
        
        
        
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        
        self.Layers = torch.nn.ModuleList([ MLSTMCell3(input_dims=self.input_dims, inner_dims_Q=[self.hidden_dims[0], 2*self.hidden_dims[1]], inner_dims_R=ExtraInput_dims, hidden_dims=self.hidden_dims) for _ in range(self.num_layers)])
                                                                                                                #  ^ as inner_dims is prev layer's H and this layer's H concatenated. 
            
        self.device = device
        
        
    def forward(self, X_t, H_prev, ExtraInput_t, C_prev):
        
        """
        X_t : input matrix of shape [batch_size, *input_dims], this is timestep t's input matrix.
        H_prev : list of length self.num_layers, each instance is a tensor of shape [batch_size, *hidden_dims]. At t=0, make sure it is just a list of 0 matrix.
        C_prev : list of length self.num_layers, each instance is a tensor of shape [batch_size, *hidden_dims]. At t=0, make sure it is just a list of 0 matrix.
        ExtraInput_t : Extra Tensor as Input of shape [batch_size, *ExtraInput_dims]. At t=0, make sure it is just a 0 matrix.


        RETURNS: 
                H_t_list : list of length self.num_layers, each instance is a tensor of shape [batch_size, *hidden_dims]
                C_t_list : list of length self.num_layers, each instance is a tensor of shape [batch_size, *hidden_dims]
        """
    #         t1 = time.time()

        H_0_t = torch.zeros([X_t.shape[0], *self.hidden_dims], device=self.device) # For layer 0 at all time.
        
        appended_H_prev_list = [H_0_t, *H_prev]
        
        C_t_list = []
        
        for i in range(1, self.num_layers+1):
                                    #            Prev time, current layer ; Current time, prev layer 
                layer_Hidden_input = torch.cat( [ appended_H_prev_list[i], appended_H_prev_list[i-1] ] , dim = 2)
                
                H_t, C_t = self.Layers[i-1]( X_t, layer_Hidden_input, ExtraInput_t ,C_prev[i-1] )
            
                appended_H_prev_list[i] = H_t
                C_t_list.append(C_t)

    #         t2 = time.time()
    #         print("MLSTM called. Time taken: {}".format(t2-t1))
        
        
        return appended_H_prev_list[1:], C_t_list
        
        

                                



        
        
        
        
#### A general function to give a count of trainable parameters.
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    