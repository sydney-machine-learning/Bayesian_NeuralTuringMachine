import MatrixNTM
import torch
import numpy as np
import matplotlib.pyplot as plt
#from IPython.display import clear_output
import sys
import os

torch.manual_seed(5239)

###############################################

def Bit_Seq_Error(output , target, batch_size, sequence_length):
    bit_output = output.detach().clone()
    bit_output[bit_output >= 0.5] = 1.0
    bit_output[bit_output != 1.0] = 0.0
    bit_error = torch.sum((torch.abs(target - bit_output)) )/(batch_size*sequence_length)
    return bit_error



#################################
NUM_EPOCHS = 200000

#ITEM_SIZE = 2 #Size of the group which you would like to call "1 Item".

BATCH_SIZE = 16
INPUT_DIMS  = [5,5]
HIDDEN_DIMS = [[30,30],[30,30],[30,30],[30,30]]  #Please keep all layers of same size, different sized layers would be supported tomorrow!!
OUTPUT_DIMS = [5,5]
NUM_SLOTS = 120
MEM_SLOT_DIMS = [10,10]
SR = 1 #Shift Range
EPS = 1e-8

LR = 1e-4

DEVICE = 'cuda:0' #'None' for CPU 

LOAD_SAVE = False

SAVE_PATH = 'SavedModels/RNNMatNTM_Copy1/Train5Noise/' #kaiming_uniform has mode = 'fan_out' and LeakyReLU is being used with negative_slope = 1e-1
#######################################

def MatrixCopyDataGen(batch_size = 32, item_size = [4,5], timesteps = 2, device = None):
    
    assert item_size[0] == item_size[1] - 1
    
    pre_content = torch.rand([batch_size, item_size[0]+1, item_size[1]*timesteps])
    
    pre_content[:,:-1,:][pre_content[:,:-1,:] > 0.5] = 1.0
    pre_content[:,-1,:] = 0.0
    pre_content[pre_content != 1] = 0.0
        
    limiter = torch.zeros([batch_size, item_size[0]+1, item_size[1]])
    limiter[:,-1,:] = 1.0
    #response_sheet = torch.zeros_like(pre_content)
    question_ = torch.cat([pre_content, limiter], dim = 2) 
    question = question_ + torch.randn(question_.shape)*1e-3#np.random.normal(loc=0,scale=1e-3,size=question_.shape)
    
    return question.to(device), pre_content.to(device)

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
    p.register_hook(lambda grad: torch.clamp(grad, -10, 10)) 
print("All gradients will be clipped between {} ".format((-10,10)))


##################################################

loss_func = torch.nn.BCELoss()

if LOAD_SAVE == True:
  optim = torch.optim.RMSprop(params = MNTM.parameters(), lr = LR, momentum = 0.9)
  optim.load_state_dict(optim_state_dict)

else:
  optim = torch.optim.RMSprop(params = MNTM.parameters(), lr = LR, momentum = 0.9)

###########################################################




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



try:

    for i in range(NUM_EPOCHS):

      if ((i+1)%500) == 0:
        _ = os.system('clear')

      num_t =  int( torch.randint(low=1, high=21, size=[], device=DEVICE) )

      inp, out = MatrixCopyDataGen(BATCH_SIZE,[INPUT_DIMS[0]-1, INPUT_DIMS[1]],num_t,DEVICE)
      
      response_sheet = torch.zeros_like(out)

      print("Epoch: ",i)
      print("--->Sequence Length: ",int(num_t))

      MNTM.HiddenReset()

      MNTM.zero_grad()

      #Feeding the sequence
      for t in range(1,num_t+2):
        _ = MNTM(inp[:,:,(t-1)*INPUT_DIMS[1]:t*INPUT_DIMS[1]])

      #Taking Output from controller now, for backprop
      output = []

      for t in range(1,num_t+1):
        MNTM_out = MNTM(response_sheet[:,:,(t-1)*INPUT_DIMS[1]:t*INPUT_DIMS[1]])
        output.append(MNTM_out)


      res = torch.cat(output, dim=2)

      loss = loss_func(res, out)
      bse = Bit_Seq_Error(res, out, BATCH_SIZE, num_t)

      print("----->Loss: {}".format(loss))
      print("----->Sequence Bit Error: {}".format(bse))

      print("--||--||--||--||--||--||--||--||--")

      loss.backward()

      optim.step()

      TrainLosses[i] = loss.detach().data
      TrainBitErrors[i] = bse

      torch.cuda.empty_cache()

      #Regular Check
      if (i+1)%100 == 0:


        # num_t = torch.randint(low=70,high=101,size=[])
        num_t = 20

        inp, out = MatrixCopyDataGen(BATCH_SIZE,[INPUT_DIMS[0]-1, INPUT_DIMS[1]],num_t,DEVICE)
        
        response_sheet = torch.zeros_like(out)

        print("Epoch: ",i)
        print("--->Sequence Length: ",int(num_t))

        MNTM.HiddenReset()

        MNTM.zero_grad()

        #Feeding the sequence
        for t in range(1,num_t+2):
            _ = MNTM(inp[:,:,(t-1)*INPUT_DIMS[1]:t*INPUT_DIMS[1]])

        #Taking Output from controller now, for backprop
        output = []

        for t in range(1,num_t+1):
            MNTM_out = MNTM(response_sheet[:,:,(t-1)*INPUT_DIMS[1]:t*INPUT_DIMS[1]])
            output.append(MNTM_out)

        res = torch.cat(output, dim=2)

        loss = loss_func(res, out)
        bse = Bit_Seq_Error(res, out, BATCH_SIZE, num_t)

        print("\n---------->Validation Loss on Sequence Length {} : {}".format(num_t,loss))
        print("---------->Validation BSE on Sequence Length {} : {}".format(num_t, bse))
        print("-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-")

        ValidationLosses[int((i+1)/100)-1] = loss.detach().data
        ValidationBitErrors[int((i+1)/100)-1] = bse


      if (i+1)%1000 == 0:
        #pass
        Save()
        torch.save({'TrainLoss' : TrainLosses, 'TrainBSE' : TrainBitErrors, 'ValidationLoss' : ValidationLosses, 'ValidationBSE' : ValidationBitErrors, 'TillEpoch' : i}, SAVE_PATH + 'LossTrace.pth')

except ValueError as e:
  #print(e)
  print("Value error... Get up: ", sys.exc_info()[0])
  Save()
  torch.save({'TrainLoss' : TrainLosses, 'TrainBSE' : TrainBitErrors, 'ValidationLoss' : ValidationLosses, 'ValidationBSE' : ValidationBitErrors, 'TillEpoch' : i},SAVE_PATH + 'LossTrace.pth')
  print("Model Save Successfull!")


