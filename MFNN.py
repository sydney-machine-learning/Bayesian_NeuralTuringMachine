import torch

class MFNN(torch.nn.Module):
    
    def __init__(self, input_dims, output_dims):
        super().__init__()
        
        assert len(input_dims) == len(output_dims) == 2
        
        self.input_dims = input_dims # [r, c]
        self.output_dims = output_dims # [r_prime, c_prime]

        
        #Parameters of the model
        self.U = torch.nn.Parameter( torch.empty([output_dims[0], input_dims[0]])  ) #Shape = [r_prime, r]
        torch.nn.init.kaiming_uniform_(self.U)
        
        self.V = torch.nn.Parameter( torch.empty([input_dims[1], output_dims[1]])  ) #Shape = [c, c_prime]
        torch.nn.init.kaiming_uniform_(self.V)
        
        self.B = torch.nn.Parameter( torch.empty(output_dims)  ) #Shape = [r_prime, c_prime]
        torch.nn.init.kaiming_uniform_(self.B)
        
        
        
    def forward(self, inp):
        
        #assert list(inp.shape)[1:] == self.input_dims, 'Wrong Shape for the input.'
        
        Y = torch.matmul(torch.matmul(self.U, inp) , self.V ) + self.B
        
        return Y

        
        
        