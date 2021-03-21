import torch
from prettytable import PrettyTable
import time

class MFNN(torch.nn.Module):
    
    def __init__(self, input_dims, output_dims, bias=True):
        super().__init__()
        
        assert len(input_dims) == len(output_dims) == 2
        
        self.input_dims = input_dims # [r, c]
        self.output_dims = output_dims # [r_prime, c_prime]
        self.bias = bias
        
        #Parameters of the model
        self.U = torch.nn.Parameter( torch.empty([output_dims[0], input_dims[0]])  ) #Shape = [r_prime, r]
        torch.nn.init.kaiming_uniform_(self.U)
        
        self.V = torch.nn.Parameter( torch.empty([input_dims[1], output_dims[1]])  ) #Shape = [c, c_prime]
        torch.nn.init.kaiming_uniform_(self.V)
        
        if self.bias:
            self.B = torch.nn.Parameter( torch.empty(output_dims)  ) #Shape = [r_prime, c_prime]
            torch.nn.init.kaiming_uniform_(self.B)
            
        
        
    def forward(self, inp):
        
        #assert list(inp.shape)[1:] == self.input_dims, 'Wrong Shape for the input.'
        
        if self.bias:
            return torch.matmul(torch.matmul(self.U, inp) , self.V ) + self.B
        
        else:
            return torch.matmul(torch.matmul(self.U, inp) , self.V )

        

        

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


class MRNNCell(torch.nn.Module):
    
    def __init__(self, input_dims, state_dims, mem_slot_dims):
        """
        input_dims : list of shape of the input matrix.
        state_dims : list of shape of the state dimesnions.
        mem_slot_dims : list of shape of the memory slot's dimension.
        """
        
        
        super().__init__()
        
        self.input_dims = input_dims
        self.state_dims = state_dims
        self.mem_slot_dims = mem_slot_dims
        
        self.Connections = MFNN3(P_input_dims=state_dims, Q_input_dims=input_dims, R_input_dims=mem_slot_dims, output_dims=state_dims)
        self.relu = torch.nn.ReLU()
        
        
        
    def forward(self, hidden_prev, X_t, read_prev):
        """
        X_t : Tensor of shape [batch_size, *input_dims].
        """
        #assert list(X_t.shape)[1:] == self.input_dims
        hidden_t = torch.tanh( self.Connections(P=hidden_prev, Q=X_t, R=read_prev) )
        
        return hidden_t



