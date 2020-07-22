import collections
import torch
import torch.nn.functional as F
import time


###   Use them in the main class.

HeadOpsOutput = collections.namedtuple('HeadOpsOutput', ('AllWeights', 'ReadWeighings', 'WriteWeighings', 'EraseMatList', 'AddMatList'))#  |


class MNTMHeadOp(torch.nn.Module):
    
    def __init__(self, Num_MemSlots, MemorySlot_dims, num_RH, num_WH, split_dict, shift_range = 1, eps = 1e-8):
        
        super().__init__()
        
        if len(MemorySlot_dims) > 2:
            raise ValueError("Currently, only 3 Dimensional Orthotopal Memory is supported.")
    
        self.N = Num_MemSlots # N slots of Memory
        self.MemorySlot_dims = MemorySlot_dims #list of dimensions of memory slot.
        self.num_RH = num_RH
        self.num_WH = num_WH
        self.shift_range = shift_range
        self.eps = eps
            
        self.M1 = MemorySlot_dims[0]
        self.M2 = MemorySlot_dims[1]
        
        total_heads = num_RH + num_WH
        self.total_heads = total_heads
           
        #self.split_name_list = ['K_t', 'A_t_and_E_t', 's_t', 'beta_g_gamma']
        
        self.split_K_t = [self.M2 for _ in range(self.total_heads)]
        self.split_A_t_E_t = [self.M2 for _ in range(2*self.num_WH)]
        self.split_s_t = [1 for _ in range(self.total_heads)]
        self.split_beta_g_gamma = [1 for _ in range(self.total_heads)]
        
        self.softplus = torch.nn.Softplus(threshold = 20)
        
        #assert (2*shift_range + 1 + 3) <= self.M2
        #We can loose this constraint due to third revision.
        
        
    def forward(self, InterfaceMatrices, Prev_Memory, PrevHeadOpsTensors):
        """
        PrevHeadOpsTensors is a namedtuple of same type as PrevTensors. It should have a list Prev_W_list which should be of length total_heads and shape of 
        each tensor [batch_size, N]
        """
        
        K_t_list = torch.split(InterfaceMatrices['K_t'], self.split_K_t, dim = 2)
        assert list(K_t_list[0].shape)[1:] == [self.M1, self.M2], '{}'.format(list(K_t_list[0].shape))
        
        AE_list = torch.split(InterfaceMatrices['A_t_and_E_t'], self.split_A_t_E_t, dim = 2)
        assert list(AE_list[0].shape)[1:] == [self.M1, self.M2], '{}'.format(list(AE_list[0].shape))
        A_t_list = AE_list[:self.num_WH]
        E_t_list = AE_list[self.num_WH:2*self.num_WH]
        
        s_t_list = torch.split(InterfaceMatrices['s_t'], self.split_s_t, dim = 2)
        assert list(s_t_list[0].shape)[1:] == [2*self.shift_range+1,1], '{}'.format(list(s_t_list[0].shape))
        
        bgg_list = torch.split(InterfaceMatrices['beta_g_gamma'], self.split_beta_g_gamma, dim = 2)
        assert list(bgg_list[0].shape)[1:] == [3,1], '{}'.format( list(bgg_list[0].shape))
        #beta_g_gamma as bgg
        
        
        
#         assert list(Interface_t.shape)[1:] == [ self.M1, 2*self.M2*self.num_WH + (self.M2 + 1)*self.total_heads ]
#         assert list(Prev_Memory.shape)[1:] == [ self.N, self.M1, self.M2 ]
        
#         parameters = torch.split(Interface_t, self.split_list, dim = 2)
        
#         K_t_list = parameters[ : self.total_heads]
#         E_t_list = parameters[self.total_heads : self.total_heads + self.num_WH]
#         A_t_list = parameters[self.total_heads + self.num_WH : self.total_heads + 2*self.num_WH]
#         Others_params_list = parameters[self.total_heads + 2*self.num_WH : 2*self.total_heads + 2*self.num_WH] # Each element of this list is a vector of length M2, where we take first 2*shift_range+1, next 1, next 1 and then the next 1 as s_t, beta_t, g_t, gamma_t respectively. If some other scalar still remains (i.e. M2 > 2*shift_range+1 + 3), we let them be.
        
#         self.K_t_list = []
        
#         self.beta_t_list = []
        
#         self.softmax_weights = []
        
#         self.W_c_t_list = []
        
#         self.g_t_list = []
        
#         self.W_g_t_list = []

#         self.gamma_t_list = []        
        
#         self.W_hat_t_list = []
        
#         self.s_t_list = []
        
#         self.W_hat_t_sharpened_list = []
        
        New_W_list = []
        
        for i in range(self.num_WH):
            
            # For erase_t
            assert list(E_t_list[i].shape)[1:] == self.MemorySlot_dims
            torch.sigmoid_(E_t_list[i])
            
            # For a_t
            assert list(A_t_list[i].shape)[1:] == self.MemorySlot_dims
            torch.tanh_(A_t_list[i])
            
            
        for i in range(self.total_heads):
            
            
            #Param_Vec = Others_params_list[i].squeeze(2)
            #assert list(Param_Vec.shape)[1:] == [self.M1]
            
            s_t = s_t_list[i].squeeze(-1)
            assert list(s_t.shape)[1:] == [2*self.shift_range+1], '{}'.format(list(s_t.shape))
            
            
            beta_t = bgg_list[i][:,0]
            assert list(beta_t.shape)[1:] == [1], '{}'.format(list(beta_t.shape))
            
            g_t = bgg_list[i][:,1]
            assert list(g_t.shape)[1:] == [1], '{}'.format(list(g_t.shape))
            
            gamma_t = bgg_list[i][:,2]
            assert list(gamma_t.shape)[1:] == [1], '{}'.format(list(gamma_t.shape))
            

            
            # For W_c_t
            
            
            K_t = K_t_list[i]
            assert list(K_t.shape)[1:] == self.MemorySlot_dims
            
            torch.tanh_(K_t)   # To bring it into (-1,1)
            
            #self.K_t_list.append(K_t)
            
            Mat_Sim = self.MatrixSimilarity(K_t, Prev_Memory, self.eps)
            #Shape : [batch_size, N]
            
            
            
            beta_t_compat = 1 + self.softplus(beta_t)
            #self.beta_t_list.append(beta_t_compat)
        
            softmax_weights = torch.mul(beta_t_compat, Mat_Sim)  # beta_t_compat*5 to increase key's strength. 
            #Shape : [batch_size, N]
            
            #self.softmax_weights.append(softmax_weights)
            
            
            exponents = torch.exp(softmax_weights.clamp(0.0, 80.0) )
            sums = (torch.sum(exponents, dim = 1) + self.eps)
            
            W_c_t = exponents / sums.unsqueeze(1)
            #self.W_c_t_list.append(W_c_t)
        
            assert list(W_c_t.shape)[1:] == [self.N]            
            
            
            
            
            
            
            # For W_g_t
            
            assert list(g_t.shape)[1:] == [1]
            
            g_t_compat = torch.sigmoid(g_t)
            #self.g_t_list.append(g_t_compat)
                       
                
            W_g_t = g_t_compat * W_c_t + (1-g_t_compat) * PrevHeadOpsTensors.Prev_W_List[i]
            #self.W_g_t_list.append(W_g_t)
            
            
            assert list(W_g_t.shape)[1:] == [self.N]
            
            
            
            
            
            
            # For W_hat_t 
            
            assert list(s_t.shape)[1:] == [2*self.shift_range+1]
            
            s_t_softmaxed = self.softmax(s_t, dim = 1, eps = self.eps) # s_t is softmax'ed
            
            W_hat_t = self.CircularConvolution(W_g_t, s_t_softmaxed, self.shift_range)
            #Shape : [batch_size, N]
            
            #self.s_t_list.append(s_t_softmaxed)
            
            #self.W_hat_t_list.append(W_hat_t)
            
            
            
            
            
            
            
            # For W_t
            
            
            assert list(gamma_t.shape)[1:] == [1]
            
            gamma_t_compat = 1 + self.softplus(gamma_t) 
            
            #self.gamma_t_list.append(gamma_t_compat)
            
            
            W_hat_t_sharpened = torch.pow(W_hat_t, gamma_t_compat)
            #Shape : [batch_size, N]
            
            #self.W_hat_t_sharpened_list.append(W_hat_t_sharpened)
            
            denom = torch.sum(W_hat_t_sharpened, dim = 1)
            
            W_t = W_hat_t_sharpened / (denom.unsqueeze(1) + self.eps)
            
            assert list(W_t.shape)[1:] == [self.N]
            
            with torch.no_grad():
                ################# Special Check, don't remove!! ######## 
                if torch.any( torch.isnan(W_t) + torch.isinf(W_t) ):
                    raise ValueError("Yo... the Head Ops turned Anti-Christ bruh. At Head {} ... ".format(i))
                

            
            New_W_list.append(W_t)
            
            
        R_weighings_list = New_W_list[:self.num_RH]
        W_weighings_list = New_W_list[self.num_RH:self.num_RH+self.num_WH]
        
#         t2 = time.time()
#         print("MNTMHeadOps called. Time taken: {}".format(t2-t1))
        
            
        return HeadOpsOutput(AllWeights = New_W_list, ReadWeighings = R_weighings_list, WriteWeighings = W_weighings_list, EraseMatList = E_t_list, AddMatList = A_t_list)
            

    def MatrixSimilarity(self, K_t, Memory, eps):
        """
        Calculates torch.sum(X*Y) / ( torch.norm(X)*torch.norm(Y) )
        
        K_t : [batch_size, *MemorySlot_dims]
        Memory : [batch_size, N, *MemorySlot_dims]
        """
        
        res1 = Memory*K_t.unsqueeze(1)
        res2 = torch.sum(res1, dim = [2,3])
        
        normK = torch.norm(K_t, dim = [1,2])
        normMem = torch.norm(Memory, dim = [2,3])
        
        denom = normMem*normK.unsqueeze(1)
        
        res = res2/(denom + eps)
        
        #assert list(res.shape)[1:] == [self.N]
        
        return res
    
    def softmax(self, tensor, dim, eps):
        """
        Calculates the softmax of the tensor along the specified dimensions.
        assumes tensor.shape[0] is the batch size.
        
        """
        
        tensor_exp = torch.exp(tensor.clamp(0.0, 80.0))
        denom = torch.sum(tensor_exp, dim=dim)
        tensor_softmax = tensor_exp / (denom.view([ tensor.shape[0],*[1 for i in range(len(tensor.shape)-1)] ]) + eps)
        
        return tensor_softmax
    
    
    def CircularConvolution(self, inp, weight, shift_range):
        """
        Performs circular convolution of weight on inp.

        Assuming inp.shape == [batch_size, N1, N2, ... , NT]
        and      weight.shape == [batch_size, W, W, ... , W (T times)]

        shift_range = self.shift_range
        """

        #assert len(weight.shape) == len(inp.shape)
        #assert inp.shape[0] == weight.shape[0]

        batch_size = inp.shape[0]

        conv_dim = len(weight.shape[1:])

        circular_padding = F.pad(inp.unsqueeze(1), pad = [shift_range]*2*conv_dim, mode = 'circular')

        x = circular_padding.permute([1,0,*[i for i in range(2,2+conv_dim)]])

        if conv_dim == 1:
            conv_out = F.conv1d(x, weight.unsqueeze(1), groups = batch_size)
        elif conv_dim == 2:
            conv_out = F.conv2d(x, weight.unsqueeze(1), groups = batch_size)
        elif conv_dim == 3:
            conv_out = F.conv3d(x, weight.unsqueeze(1), groups = batch_size)

        result = conv_out.squeeze(0)

        assert list(result.shape) == [batch_size,*list(inp.shape[1:])]

        return result