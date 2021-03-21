import collections
import torch 
import MRNN
from MRNN import MFNN
from prettytable import PrettyTable
import time

"""
A Complete Matrix Controller for any type (hopefully) of Matrix Memory Augmented Model.

Input on the forward pass: Either ControllerInput_w_PRM or ControllerInput_wo_PRM.
Output on the forward pass: ControllerOutput.


"""

# ControllerInput = collections.namedtuple('ControllerInput', ['InputMatrix', 'HiddenStatesList_prev', 'CellStatesList_prev', 'ReadMatList_prev', 'HeadOpsTensors_prev']) #w_PRM ---> with Previours Read Matrices. Input container for NTM, DNC ....
ControllerInput = collections.namedtuple('ControllerInput',['InputMatrix', 'HiddenState_prev', 'ReadMat_prev', 'HeadOpsTensors_prev'])

ControllerOutput = collections.namedtuple('ControllerOutput', ['FinalOutput', 'ReadMat_t', 'HiddenState_t', 'NewHeadOpsTensors_prev']) # Output container for the Controller, thus for the whole MANN.


# LSTMInput = collections.namedtuple('LSTMInput', ['InputMatrix', 'HiddenStatesList_prev', 'CellStatesList_prev'])
# LSTM_States = collections.namedtuple('LSTM_States', ['HiddenStatesList', 'CellStatesList'])

#                                                                                                     ---------------
#                                                                                                     |             |   
#HeadOpsInput = collections.namedtuple('HeadOpsInput', ('InterfaceMatrices', 'Memory_Prev', 'HeadOpsTensors_prev')) # ----------------------
HeadOpsOutput = collections.namedtuple('HeadOpsOutput', ('AllWeights', 'ReadWeighings', 'WriteWeighings', 'EraseMatList', 'AddMatList'))#  |
HeadOpsTensors_prev = collections.namedtuple('HeadOpsTensors_prev', ['Prev_W_List']) #               <--------------------------------------


class Controller(torch.nn.Module):

	def __init__(self, input_dims, hidden_dims, num_layers, output_dims, HEADOPS , MEMORYUNIT , NumMemSlots, MemSlotDims, split_dict, device=None):

		super().__init__()

		self.input_dims = input_dims
		self.output_dims = output_dims
		self.hidden_dims = hidden_dims
		self.num_layers = num_layers
		for i in range(1, self.num_layers):
			assert hidden_dims[0] == hidden_dims[i] 

		# self.num_RH = num_RH
		# self.num_WH = num_WH  # => Kept 1 in the experiments.
 
		self.NumMemSlots = NumMemSlots
		self.MemSlotDims = MemSlotDims

		self.split_dict = split_dict

		self.device = device

		self.HEADOPS = HEADOPS # Instance of HeadOps class.
		self.MEMORYUNIT = MEMORYUNIT # Instance of MemoryUnit class.
		self.RNNLayerStates = [0 for _ in range(self.num_layers)]

		# self.MLSTM = MLSTM.MLSTM3(input_dims=self.input_dims, hidden_dims=self.hidden_dims, ExtraInput_dims=[MemSlotDims[0], MemSlotDims[1]*num_RH], num_layers=self.num_layers, device=device)
		self.RNN = torch.nn.ModuleList([MRNN.MRNNCell(input_dims=self.input_dims, state_dims= hidden_dims[0], mem_slot_dims=MemSlotDims),*[MRNN.MRNNCell(input_dims=self.input_dims, state_dims= hidden_dims[i], mem_slot_dims=MemSlotDims) for i in range(1,self.num_layers)]])

		# self.LinearY = MFNN(input_dims=[self.hidden_dims[0], self.num_layers*self.hidden_dims[1]], output_dims=self.output_dims)
		self.LinearY = MFNN(input_dims = hidden_dims[-1], output_dims = output_dims)        
		
		# self.LinearR = MFNN(input_dims=[ self.MemSlotDims[0], self.MemSlotDims[1]*self.num_RH ], output_dims=self.output_dims) #To weigh the currently Read Elements from memory.
		self.LinearR = MFNN(input_dims = MemSlotDims, output_dims = output_dims)

		IntefaceElements = {}
		for k,v in self.split_dict.items():
		    # IntefaceElements[k] = MFNN(input_dims=[self.hidden_dims[0], self.num_layers*self.hidden_dims[1]], output_dims=v)            
		    IntefaceElements[k] = MFNN(input_dims = hidden_dims[0], output_dims = v)

		self.InterfaceLayers = torch.nn.ModuleDict(IntefaceElements)
		#This dictionary of Linear layers will convert the Hidden states to respective Matrices of name and shape as specified by split-dict.

		self.LinearHidden = MFNN(input_dims = [self.hidden_dims[0][0], self.num_layers*self.hidden_dims[0][1]], output_dims = self.hidden_dims[0])
		#Above converts the hidden states of each layer to a hidden matrix.


	def forward(self, ControllerInput):



		# HiddenStatesList_t, CellStatesList_t = self.MLSTM( ControllerInput.InputMatrix  ,  ControllerInput.HiddenStatesList_prev  ,  torch.cat(ControllerInput.ReadMatList_prev, dim = 2)  ,  ControllerInput.CellStatesList_prev )
		self.RNNLayerStates[0] = self.RNN[0](hidden_prev = ControllerInput.HiddenState_prev[0], X_t = ControllerInput.InputMatrix, read_prev = ControllerInput.ReadMat_prev, hidden_layer_prev = torch.zeros_like(ControllerInput.HiddenState_prev[0]))
		for i in range(1,self.num_layers):
		    self.RNNLayerStates[i] = self.RNN[i](hidden_prev = ControllerInput.HiddenState_prev[i], X_t = ControllerInput.InputMatrix, read_prev = ControllerInput.ReadMat_prev, hidden_layer_prev = self.RNNLayerStates[i-1])
		HiddenState_t = self.LinearHidden(torch.cat(self.RNNLayerStates, dim=2))
		#HiddenState_t = self.RNN(hidden_prev = ControllerInput.HiddenState_prev, X_t = ControllerInput.InputMatrix, read_prev = ControllerInput.ReadMat_prev)

		# HiddenStatesTensor = torch.cat(HiddenStatesList_t, dim = 2)
		# assert list(HiddenStatesTensor.shape)[1:] == [self.hidden_dims[0], self.hidden_dims[1]*self.num_layers]


		# Generating Controller Head's Emmision.
		InterfaceMatrices = {}
		for k in self.InterfaceLayers.keys():
		    InterfaceMatrices[k] = self.InterfaceLayers[k](HiddenState_t)


		HO_Out = self.HEADOPS(InterfaceMatrices, self.MEMORYUNIT.Memory, ControllerInput.HeadOpsTensors_prev)
		#HeadOps Output, namedtuple

		self.HO_Out = HO_Out #For Visualisation purposes.
 
		NewHeadOpsTensors_prev = HeadOpsTensors_prev(Prev_W_List = HO_Out.AllWeights)
		#HeadOpsTensors_prev for next timestep. Returned in ControllerOutput.

		ReadMat_t = self.MEMORYUNIT(HO_Out)[0] #It's anyways a list of length 1 in this case.

		# ReadMatTensor_t = torch.cat(ReadMatList_t, dim = 2)
		# assert list(ReadMatTensor_t.shape)[1:] == [self.MemSlotDims[0], self.MemSlotDims[1]*self.num_RH], '{}'.format(ReadMatTensor_t.shape)


		##################################### Calculating Output Now.
		V_t = self.LinearY(HiddenState_t)

		Y_t = torch.sigmoid(V_t + self.LinearR(ReadMat_t))

		ControlOutput = ControllerOutput(FinalOutput = Y_t, ReadMat_t = ReadMat_t, HiddenState_t = self.RNNLayerStates, NewHeadOpsTensors_prev = NewHeadOpsTensors_prev)

		return ControlOutput
