import MatrixNTM
import torch
import numpy as np
import matplotlib.pyplot as plt
#from IPython.display import clear_output
import sys
import os

torch.manual_seed(1482)

###############################################

def Bit_Seq_Error(output , target, batch_size, sequence_length):
    bit_output = output.detach().clone()
    bit_output[bit_output >= 0.5] = 1.0
    bit_output[bit_output != 1.0] = 0.0
    bit_error = torch.sum((torch.abs(target - bit_output)) )/(batch_size*sequence_length)
    return bit_error



#################################
NUM_EPOCHS = 200000

ITEM_SIZE = 2 #Size of the group which you would like to call "1 Item".

BATCH_SIZE = 16
INPUT_DIMS  = [5,5]
HIDDEN_DIMS = [[20,20],[20,20],[20,20],[20,20]]  #Please keep all layers of same size, different sized layers would be supported tomorrow!!
OUTPUT_DIMS = [5,5]
NUM_SLOTS = 120
MEM_SLOT_DIMS = [6,6]
SR = 1 #Shift Range
EPS = 1e-9

LR = 8e-5

DEVICE = 'cuda:0' #'None' for CPU 

LOAD_SAVE = False

SAVE_PATH = 'SavedModels/RNNMatNTM_AssRec1/Train6Duplicate/'
#######################################

def OneBatchSequence(seq_size = [4,5], item_size = 2, num_items = 15, query_num = 14):
    
    assert num_items > query_num
    
    limiter = torch.zeros([seq_size[0]+1, seq_size[1]])
    limiter[-1,0] = 1.0
    limiter[-1,-1] = 1.0
    
    
    cat_list = []
    for i in range(num_items):
        
        cat_list.append(limiter)
        
        content = torch.rand([seq_size[0]+1, seq_size[1] * item_size])
        content[:-1,:][content[:-1,:] > 0.5] = 1.0
        content[-1,:] = 0.0
        content[content != 1] = 0.0
        cat_list.append(content)
        
    
    delimeter = torch.zeros([seq_size[0]+1, seq_size[1]])
    delimeter[-1,1:-1] = 1.0
    cat_list.append(delimeter)
    
    
    start = seq_size[1]*query_num + (query_num - 1)*(seq_size[1]*2)
    stop = seq_size[1]*query_num + (query_num - 1)*(seq_size[1]*2) + (seq_size[1]*2) 
    
    sequence = torch.cat(cat_list, dim = 1)
    
    query = sequence[:,start:stop]
    expected_result = sequence[:,start+3*seq_size[1] : stop+3*seq_size[1]]

    
    finalsequence = torch.cat([sequence, query], dim = 1)
            
    
    return finalsequence, expected_result




def GenRandBatchSeq(batch_size=32, seq_size = [4,5], item_size = 2, num_items = 15, device = None):
    
    seq_list = []
    res_list = []
    
    
    
    for i in range(batch_size):
        query_num = torch.randint(low=1, high=num_items, size=[])
        seq, res = OneBatchSequence(seq_size=seq_size, item_size=item_size, num_items=num_items, query_num=query_num)
        seq_list.append(seq.unsqueeze(0))
        res_list.append(res.unsqueeze(0))
    
    seq_tensor = torch.cat(seq_list, dim=0)
    res_tensor = torch.cat(res_list, dim=0)
    
    return seq_tensor.to(device), res_tensor.to(device)
#####################################################################################


if LOAD_SAVE == True:
  MNTM = torch.load(SAVE_PATH + 'MNTM.pth')
  optim_state_dict = torch.load(SAVE_PATH + 'optim_state_dict.pth')
  LossTrace = torch.load(SAVE_PATH + 'LossTrace.pth')
  print("Previously Trained till : {}".format(LossTrace['TillEpoch']))
  if LossTrace['TillEpoch'] == NUM_EPOCHS-1:
    print("Training was completed previously!! {} ".format(LossTrace['TillEpoch']))

else:
  MNTM = MatrixNTM.MatRNN_NTM(batch_size=BATCH_SIZE, input_dims=INPUT_DIMS, hidden_dims=HIDDEN_DIMS, output_dims=OUTPUT_DIMS, NumSlots=NUM_SLOTS, MemorySlot_dims=MEM_SLOT_DIMS, shift_range=SR, eps = EPS, device=DEVICE)
  MNTM = MNTM.to(DEVICE)

MNTM.count_parameters()

#####################################################################################


def Save(optim_dict_too = True):
	torch.save(MNTM, SAVE_PATH + 'MNTM.pth')
	if optim_dict_too:
		torch.save(optim.state_dict(), SAVE_PATH + 'optim_state_dict.pth')

##################################################################################

#Backward Hooks for gradient clipping
for p in MNTM.parameters():
    p.register_hook(lambda grad: torch.clamp(grad, -20, 20)) 
print("All gradients will be clipped between {} ".format((-20,20)))


##################################################

loss_func = torch.nn.BCELoss()

if LOAD_SAVE == True:
  optim = torch.optim.RMSprop(params = MNTM.parameters(), lr = LR, momentum = 0.9)
  optim.load_state_dict(optim_state_dict)

else:
  optim = torch.optim.RMSprop(params = MNTM.parameters(), lr = LR, momentum = 0.9)

###########################################################


min_loss = 0.7

if LOAD_SAVE == False:
  a = [0.0 for _ in range(NUM_EPOCHS)]
  b = [0.0 for _ in range(NUM_EPOCHS)]
  c = [0.0 for _ in range(int(NUM_EPOCHS/100))]
  d = [0.0 for _ in range(int(NUM_EPOCHS/100))]

  TrainLosses = torch.tensor(a,dtype=torch.float64)
  TrainBitErrors = torch.tensor(b, dtype=torch.float64)

  ValidationLosses = torch.tensor(c,dtype=torch.float64)
  ValidationBitErrors = torch.tensor(d, dtype=torch.float64)

else: 
  l = LossTrace['TillEpoch'] +1
  TrainLosses = torch.zeros(NUM_EPOCHS)
  TrainLosses[:l] = LossTrace['TrainLoss']
    
  TrainBitErrors = torch.zeros(NUM_EPOCHS)
  TrainBitErrors[:l] = LossTrace['TrainBSE']


  ValidationLosses = torch.zeros(int(NUM_EPOCHS/100))
  ValidationLosses[:int(l/100)] = LossTrace['ValidationLoss']

  ValidationBitErrors = torch.zeros(int(NUM_EPOCHS/100))
  ValidationBitErrors[:int(l/100)] = LossTrace['ValidationBSE']


if LOAD_SAVE:
  start = LossTrace['TillEpoch']
else:
  start = -1


#try:

for i in range(start+1, NUM_EPOCHS):

	if ((i+1)%500) == 0:
	  _ = os.system('clear')  
   	  #clear_output(wait=False)
        

	num_items = torch.randint(low=2, high=10, size=[],device=DEVICE)

	inp, out = GenRandBatchSeq(batch_size=BATCH_SIZE, seq_size=[INPUT_DIMS[0]-1,INPUT_DIMS[1]], item_size=ITEM_SIZE, num_items=num_items, device=DEVICE)
	#inp, out = MatrixCopyDataGen(32,[4,5],num_t)

	response_sheet = torch.zeros_like(out)

	print("Epoch: ",i)

	print("--->Number of Items: ",int(num_items))

	MNTM.HiddenReset()

	MNTM.zero_grad()



	#Feeding the sequence
	#Input sequence is of length inp.shape[2]/5 

	num_sequences = int(inp.shape[2]/INPUT_DIMS[1])


	for t in range(1,num_sequences+1):
	    _ = MNTM(inp[:,:,(t-1)*INPUT_DIMS[1]:t*INPUT_DIMS[1]])
	                                #5 is the seq_size[1]                    

	#Taking Output from controller now, for backprop
	output = []
	                # 2 is the item_size
	for t in range(1,ITEM_SIZE+1):
	  MNTM_out = MNTM(response_sheet[:,:,(t-1)*INPUT_DIMS[1]:t*INPUT_DIMS[1]])
	  output.append(MNTM_out)


	res = torch.cat(output, dim=2)

	loss = loss_func(res, out)

	bse = Bit_Seq_Error(res, out, BATCH_SIZE, ITEM_SIZE)

	print("----->Loss: {}".format(loss))
	print("----->Bit Sequence Error : {}".format(bse))
	print("--||--||--||--||--||--||--||--||--")

	loss.backward()

	optim.step()

	# if LOAD_SAVE:
	#   LossTrace['TrainLoss'][i] = loss.detach().data
	#   LossTrace['TrainBSE'][i] = bse

	#else:
	TrainLosses[i] = loss.detach().data
	TrainBitErrors[i] = bse

	torch.cuda.empty_cache()

	#Regular Check
	if (i+1)%100 == 0:

	  # num_t = torch.randint(low=70,high=101,size=[])
	  num_items = 12

	  inp, out = GenRandBatchSeq(batch_size=BATCH_SIZE, seq_size=[INPUT_DIMS[0]-1,INPUT_DIMS[1]], item_size=ITEM_SIZE, num_items=num_items,  device=DEVICE)
	  #inp, out = MatrixCopyDataGen(32,[4,5],num_t)

	  response_sheet = torch.zeros_like(out)

	  print("--->Number of Items: ",int(num_items))

	  MNTM.HiddenReset()

	  MNTM.zero_grad()

	  #Feeding the sequence
	  #Input sequence is of length inp.shape[2]/5 

	  num_sequences = int(inp.shape[2]/INPUT_DIMS[1])
	                              #5 is the seq_size[1] 

	  for t in range(1,num_sequences+1):
	    _ = MNTM(inp[:,:,(t-1)*INPUT_DIMS[1]:t*INPUT_DIMS[1]])
	                                  #5 is the seq_size[1]                    

	  #Taking Output from controller now, for backprop
	  output = []
	                  # 2 is the item_size
	  for t in range(1,ITEM_SIZE+1):
	    MNTM_out = MNTM(response_sheet[:,:,(t-1)*INPUT_DIMS[1]:t*INPUT_DIMS[1]])
	    output.append(MNTM_out)                     #5 is the seq_size[1] 


	  res = torch.cat(output, dim=2)

	  loss = loss_func(res, out)

	  bse = Bit_Seq_Error(res, out, BATCH_SIZE, ITEM_SIZE)

	  print("\n---------->Validation Loss on Number of items {} : {}\n".format(num_items,loss))
	  print("\n---------->Validation BSE on Number of items {} : {}\n".format(num_items,bse))
	  print("-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-")

	  # if LOAD_SAVE:
	  #   LossTrace['ValidationLoss'][int((i+1)/100) - 1] = loss.detach().data
	  #   LossTrace['ValidationBSE'][int((i+1)/100) - 1] = bse
	  #else:
	  ValidationLosses[int((i+1)/100) - 1] = loss.detach().data
	  ValidationBitErrors[int((i+1)/100) - 1] = bse


	  if loss < min_loss:
	    min_loss = loss.detach().data
	    Save()
	    torch.save({'TrainLoss' : TrainLosses, 'TrainBSE' : TrainBitErrors, 'ValidationLoss' : ValidationLosses, 'ValidationBSE' : ValidationBitErrors, 'TillEpoch' : i}, SAVE_PATH + 'LossTrace.pth')


	  torch.cuda.empty_cache()


	if (i+1)%1000 == 0:
	  torch.save({'TrainLoss' : TrainLosses, 'TrainBSE' : TrainBitErrors, 'ValidationLoss' : ValidationLosses, 'ValidationBSE' : ValidationBitErrors, 'TillEpoch' : i}, SAVE_PATH + 'LossTrace.pth')
	  Save()
	    
# except :
#   print(" error... Get up: ", sys.exc_info()[0])
#   Save()
#   torch.save({'TrainLoss' : TrainLosses, 'TrainBSE' : TrainBitErrors, 'ValidationLoss' : ValidationLosses, 'ValidationBSE' : ValidationBitErrors, 'TillEpoch' : i}, SAVE_PATH + 'LossTrace.pth')



