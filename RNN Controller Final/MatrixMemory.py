import torch
from BatchTensorDot import btd
import time

class MatrixMemory(torch.nn.Module):
    
    def __init__(self, batch_size, MemorySlot_dims, NumSlots, num_WH, num_RH, device=None):
        
        super().__init__()
        self.batch_size = batch_size
        self.MemorySlot_dims = MemorySlot_dims
        self.N = NumSlots
        self.num_RH = num_RH
        self.num_WH = num_WH
        self.device = device
        
        
    def Init_Reset_Memory(self, scheme = 'small_const', interval = (-0.001,1e-5)):
        """
        Initializes or Resets the self.Memory which is a tensor of shape Mem_dims according to the initialization 'scheme'.
        
        Should be called first after instancing the class.
        Trivially, this should also be called after each batch of sequences has been processed.
        
        Available schemes: 1) 'small_const': Constant value taken as 'interval[1]'.
                           2) 'uniform' : Uniform Random values from the 'interval'.
        """
        
        if scheme == 'small_const':
            self.Memory = torch.ones([self.batch_size, self.N, *self.MemorySlot_dims], device=self.device)*(interval[1])
        
        elif scheme == 'uniform':
            self.Memory = torch.rand([self.batch_size, self.N, *self.MemorySlot_dims], device=self.device)*(interval[1] - interval[0]) + interval[0] 
         
        else:
            raise ValueError("Scheme '{}' has not been defined yet.".format(scheme))
        
        
        self.init_scheme = scheme 
        self.interval = interval
        
        
    def Reading(self, ReadWeighings):
        """
        Readweighings is a list of length num_RH where each element is a Tensor of shape [ batch_size, N ]. 
        
        Returns: A List of length num_RH where each element is a Tensor of shape [ batch_size, M1, M2 ] which is the Read Matrix read by that particular Read Head.  
     
        This function reads from the Memory.
        """

        #Reading from Memory.
        
        ReadMatList = []
        
        for i,READ_t in enumerate(ReadWeighings):
            
         
            #assert list(READ_t.shape) == [self.batch_size, self.N] 
            
            read_mat = btd( READ_t.unsqueeze(1) , self.Memory , d=1 ).squeeze(1)
            
            #assert list(read_mat.shape) == [self.batch_size, *self.MemorySlot_dims]
            
            ReadMatList.append(read_mat)
            
        return ReadMatList
    
    
    def Writing(self, WriteWeighings, EraseList, AddList):
        """
        Updates the Memory.
        
        WriteWeighings is a list of length num_WH, where each element is a Tensor of shape [ batch_size, N ]. 
        
        EraseList (and AddList) is a list of length num_WH, where each element is a Tensor of shape [ batch_size, M1, M2 ]
        
        REMEMBER FOR DNC, num_WH SHOULD BE 1.
        
        This function write to the memory inplace.
        """
        
        #Writing to Memory:
        
        for i,WRITE_t in enumerate(WriteWeighings):
            
            #assert list(WRITE_t.shape) == [self.batch_size, self.N]        
        
            self.Memory = self.Memory * (1 - btd( WRITE_t.unsqueeze(-1), EraseList[i].unsqueeze(1), d=1 )) + btd( WRITE_t.unsqueeze(-1) , AddList[i].unsqueeze(1) , d=1)
            
        #At this point, the Memory has been updated/written to (inplace) by the weighings WRITE_t emitted by Write Heads.

        
        
    def forward(self, HeadOpsOut):
        """
        One forward pass of Updating and Reading from Memory.
        
        HeadOpsOut is a namedtuple with fields 'WriteWeighings' (a list of length num_WH), 'ReadWeighings' (a list of length num_RH), 'AddMatList' (a list of length num_WH) and  'EraseMatList' (a list of length num_WH)
        
        Returns : A list of length num_RH where each element is a Tensor of shape [batch_size , M1, M2] which are the read matrices from each read head.
        """
#         t1 = time.time()
        
        #First, updating or writing to the memory unit:
        self.Writing(HeadOpsOut.WriteWeighings, HeadOpsOut.EraseMatList, HeadOpsOut.AddMatList)
        
        #Now, we read from the memory unit:
        ReadMatList = self.Reading(HeadOpsOut.ReadWeighings)
        
#         t2 = time.time()
#         print("MatrixMemory called. Time taken: {}".format(t2-t1))
        
        
        return ReadMatList