import torch

chars = 'abcdefghijklmnopqrstuvwxyz'

def btd(x,y,d):
        """
        Conducts batch tensor dot of x and y by contracting along last d dims of x and first d dims of y (excluding the batch size dimension)

        Assume, shape of x = [B, N1, N2, ... , NT, M1, M2, ... , MD ]
                shape of y = [B, M1, M2, ... , MD, O1, O2, ... , OR]

                where D=d.
        """
        x_shape = list(x.shape)
        y_shape = list(y.shape)
        num_x_dims , num_y_dims = len(x_shape), len(y_shape)
        
        assert (d>0) and (d<min(num_x_dims, num_y_dims)), 'Bad Value of "d" in batch tensordot'

        assert x_shape[-d:] == y_shape[1:d+1] , 'Shape of Tensors not aligned for batch tensordot.'

        x_string = chars[:num_x_dims]
        y_string = chars[num_x_dims-d:num_x_dims-d+num_y_dims-1]

        res_string = x_string[:-d]+y_string[d:]
        
        einsum_str = x_string+',' + chars[0]+y_string + '->' + res_string
    
        
        return torch.einsum(einsum_str,x,y)