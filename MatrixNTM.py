import torch
from MatrixCompleteController import Controller, ControllerInput, ControllerOutput, HeadOpsOutput, HeadOpsTensors_prev, LSTM_States
import MNTMHeadOps
import MatrixMemory
import collections

from prettytable import PrettyTable





class MatNTM(torch.nn.Module):
    
    def __init__(self, batch_size, input_dims, hidden_dims, num_layers, output_dims, num_RH, num_WH, NumSlots, MemorySlot_dims, shift_range, eps = 1e-10, device=None):
        """
        Matrix Neural Turing Machine.
        
        Here, the memory is a 3-orthotope with linear slots holding matrix representations.
        
        shift_range : (integer) the size of shift field.
                    if shift_range = 1, then head would shift left or right one step.
                    if shift_range = 2, then head would shift left, right, two steps.
                    and so on for shift_range = 3,4 ...
        """
        super().__init__()
        
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.output_dims = output_dims
        self.num_RH = num_RH
        self.num_WH = num_WH
        self.device = device
        
        self.NumSlots = NumSlots
        self.MemorySlot_dims = MemorySlot_dims
        
        self.shift_range = shift_range
        self.eps = eps
        
        self.total_heads = num_RH + num_WH
                
        
        self.M1 = self.MemorySlot_dims[0]
        self.M2 = self.MemorySlot_dims[1]
        
        self.split_dict = {'K_t' : [self.M1, self.total_heads * self.M2], 'A_t_and_E_t': [self.M1, self.num_WH*2*self.M2], 's_t' : [2*self.shift_range + 1, self.total_heads], 'beta_g_gamma' : [3,self.total_heads]}
        



        #Initializing HeadOps
        self.HeadOps = MNTMHeadOps.MNTMHeadOp(self.NumSlots, self.MemorySlot_dims, self.num_RH, self.num_WH, self.split_dict, self.shift_range, self.eps)
        
        #Initializing orthopic memory
        self.MemoryUnit = MatrixMemory.MatrixMemory(self.batch_size, self.MemorySlot_dims, self.NumSlots, self.num_WH, self.num_RH, self.device)
        self.MemoryUnit.Init_Reset_Memory(scheme = 'small_const', interval = (0,0.0001))

         #Initializing Controller
        self.Controller = Controller(input_dims=self.input_dims, hidden_dims=self.hidden_dims, num_layers=self.num_layers, output_dims=self.output_dims, num_RH=self.num_RH, num_WH=self.num_WH, HEADOPS=self.HeadOps , MEMORYUNIT=self.MemoryUnit , NumMemSlots = NumSlots, MemSlotDims = MemorySlot_dims, split_dict = self.split_dict, device=self.device)
        
        



        # Initial States of the controller at t=0.
        self.LSTM_states = LSTM_States(HiddenStatesList= [torch.zeros([self.batch_size, *self.hidden_dims], device=self.device) for _ in range(self.num_layers)], CellStatesList=[torch.zeros([self.batch_size, *self.hidden_dims], device=self.device) for _ in range(self.num_layers)] )
        
        
        # Initial Read Matrices.
        self.Prev_Read_Matrices = [ torch.zeros([self.batch_size, *self.MemorySlot_dims], device=self.device) for _ in range(self.num_RH) ]
        
        
        # Initial HeadOps Tensors.
        self.HeadOpsTensors_prev = HeadOpsTensors_prev(Prev_W_List = [self.softmax(torch.randn([self.batch_size, self.NumSlots], device=self.device)/(0.25**0.5), 1, self.eps) for _ in range(self.total_heads)])
        
        
        
    def forward(self, X_t):
        """
        X_t is a tensor of shape [batch_size, *input_dims]
        """
        
        ControlInput = ControllerInput(InputMatrix = X_t, HiddenStatesList_prev = self.LSTM_states.HiddenStatesList, CellStatesList_prev = self.LSTM_states.CellStatesList, ReadMatList_prev = self.Prev_Read_Matrices, HeadOpsTensors_prev = self.HeadOpsTensors_prev)
        ControlOutput = self.Controller(ControlInput)

        Y_t = ControlOutput.FinalOutput

        self.Prev_Read_Matrices = ControlOutput.ReadMatList_t
        self.LSTM_states = LSTM_States(HiddenStatesList = ControlOutput.HiddenStatesList_t, CellStatesList = ControlOutput.CellStatesList_t)
        self.HeadOpsTensors_prev = ControlOutput.NewHeadOpsTensors_prev

        return Y_t
        
        
    def HiddenReset(self):
        
        # Initial States of the controller at t=0.
        self.LSTM_states = LSTM_States(HiddenStatesList = [torch.zeros([self.batch_size, *self.hidden_dims], device=self.device) for _ in range(self.num_layers)], CellStatesList=[torch.zeros([self.batch_size, *self.hidden_dims], device=self.device) for _ in range(self.num_layers)] )
        
        
        # Initial Read Matrices.
        self.Prev_Read_Matrices = [ torch.zeros([self.batch_size, *self.MemorySlot_dims], device=self.device) for _ in range(self.num_RH) ]
        
        
        # Initial HeadOps Tensors.
        self.HeadOpsTensors_prev = HeadOpsTensors_prev(Prev_W_List = [self.softmax(torch.randn([self.batch_size, self.NumSlots], device=self.device)/(0.25**0.5), 1, self.eps) for _ in range(self.total_heads)])
        
        self.MemoryUnit.Init_Reset_Memory()
    
        
    def softmax(self, tensor, dim, eps):
        """
        Calculates the softmax of the tensor along the specified dimensions.
        assumes tensor.shape[0] is the batch size.
        
        """
        
        tensor_exp = torch.exp(tensor)
        denom = torch.sum(tensor_exp, dim=dim)
        tensor_softmax = tensor_exp / (denom.view([ tensor.shape[0],*[1 for i in range(len(tensor.shape)-1)] ]) + eps)
        
        return tensor_softmax


    def count_parameters(self):

	    table = PrettyTable(["Modules", "Parameters"])
	    total_params = 0
	    for name, parameter in self.named_parameters():
	        if not parameter.requires_grad: continue
	        param = parameter.numel()
	        table.add_row([name, param])
	        total_params+=param
	    print(table)
	    print(f"Total Trainable Params: {total_params}")
	    return total_params

