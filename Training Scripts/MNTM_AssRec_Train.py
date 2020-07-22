import time
import torch
import numpy as np
import MatrixNTM
from torch.utils.tensorboard import SummaryWriter



#Setting up anomaly detector (Only use during debugging)
# torch.autograd.set_detect_anomaly(mode=True)
# print("Debugging mode, anomaly detector activated!!")




torch.manual_seed(77)




# print("Waiting for previous execution to be over...")
# time.sleep(2100)




# nDNTM = torch.load('SavedModels/NTM/RUN3_MD_120_20_CE_Copy50000/nDNTM.pth')
# optim_state_dict = torch.load('SavedModels/NTM/RUN3_MD_120_20_CE_Copy50000/optim_state_dict.pth')


MNTM = MatrixNTM.MatNTM(batch_size=32, input_dims=[5,5], hidden_dims=[10,10], num_layers=3, output_dims=[5,5], num_RH=1, num_WH=1, NumSlots=120, MemorySlot_dims=[6,6], shift_range=1, device='cuda:0')
MNTM = MNTM.to('cuda:0')
print('Training on GPU .... ')
print("Training started for MEM_DIMS: {}".format([120,6,6]))                                                  #Change it on next line too!!       
print("Matrix Associative Recall Task : No Beta Amplification : 120,6,6 : RMSprop : 1e-4 : 0.9 Momentum : 0.0 Weight Decay: 5,5 : 30,30 : 3 Layers  ")





writer = SummaryWriter()





#nDNTM.CompleteReset()





loss_func = torch.nn.BCELoss()
optim = torch.optim.RMSprop(params = MNTM.parameters(), lr = 1e-4, momentum = 0.9)
#optim.load_state_dict(optim_state_dict)




# Backward Hooks for gradient clipping
# for p in MNTM.parameters():
#     p.register_hook(lambda grad: torch.clamp(grad, -10, 10)) 
# print("All gradients will be clipped between {} ".format((-10,10)))





def Save(optim_dict_too = True):
	torch.save(MNTM, 'SavedModels/MatNTM_MatAssRec2/INP5_5_HID30_30_MD80_6_6_Adam4CE_AssRec50000/MNTM.pth')
	if optim_dict_too:
		torch.save(optim.state_dict(), 'SavedModels/MatNTM_MatAssRec2/INP5_5_HID30_30_MD80_6_6_Adam4CE_AssRec50000/optim_state_dict.pth')





num_epochs = 80000






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




def Bit_Seq_Error(output , target):
    bit_output = output.detach().clone()
    bit_output[bit_output >= 0.5] = 1.0
    bit_output[bit_output != 1.0] = 0.0

    bit_error = torch.sum(torch.abs(target - bit_output))

    return bit_error



min_loss = 0.7


try:
    
    for i in range(num_epochs):

        num_items = torch.randint(low=2, high=16, size=[])

        inp, out = GenRandBatchSeq(batch_size=32, seq_size=[4,5], item_size=2, num_items=num_items, device='cuda:0')
        #inp, out = MatrixCopyDataGen(32,[4,5],num_t)

        response_sheet = torch.zeros_like(out)

        print("Epoch: ",i)

        print("--->Number of Items: ",int(num_items))

        MNTM.HiddenReset()

        MNTM.zero_grad()



        #Feeding the sequence
        #Input sequence is of length inp.shape[2]/5 

        num_sequences = int(inp.shape[2]/5)


        for t in range(1,num_sequences+1):
            _ = MNTM(inp[:,:,(t-1)*5:t*5])
                                       #5 is the seq_size[1]                    

        #Taking Output from controller now, for backprop
        output = []
                        # 2 is the item_size
        for t in range(1,2+1):
            MNTM_out = MNTM(response_sheet[:,:,(t-1)*5:t*5])
            output.append(MNTM_out)


        res = torch.cat(output, dim=2)

        loss = loss_func(res, out)

        writer.add_scalar('MatNTM_Train_MatAssRec2/INP5_5_HID30_30_MD80_6_6_Adam4CE_AssRec50000',loss,i)

        print("----->Loss: {}".format(loss))

        seq_error = Bit_Seq_Error(res, out)
        print("----->Bit Sequence Error : {}".format(Bit_Seq_Error(res, out)))

        writer.add_scalar('MatNTM_Train_MatAssRec2/Training Mean Bit Error',seq_error,i)


        loss.backward()

        #Gradients clipped...
        torch.nn.utils.clip_grad_value_(MNTM.parameters(), clip_value=20)

        optim.step()

        torch.cuda.empty_cache()

        #Regular Check
        if (i+1)%100 == 0:

            # num_t = torch.randint(low=70,high=101,size=[])
            num_items = 30

            inp, out = GenRandBatchSeq(batch_size=32, seq_size=[4,5], item_size=2, num_items=num_items,  device='cuda:0')
            #inp, out = MatrixCopyDataGen(32,[4,5],num_t)

            response_sheet = torch.zeros_like(out)

            print("--->Number of Items: ",int(num_items))

            MNTM.HiddenReset()

            MNTM.zero_grad()



            #Feeding the sequence
            #Input sequence is of length inp.shape[2]/5 

            num_sequences = int(inp.shape[2]/5)
                                        #5 is the seq_size[1] 

            for t in range(1,num_sequences+1):
                _ = MNTM(inp[:,:,(t-1)*5:t*5])
                                           #5 is the seq_size[1]                    

            #Taking Output from controller now, for backprop
            output = []
                            # 2 is the item_size
            for t in range(1,2+1):
                MNTM_out = MNTM(response_sheet[:,:,(t-1)*5:t*5])
                output.append(MNTM_out)                     #5 is the seq_size[1] 


            res = torch.cat(output, dim=2)

            loss = loss_func(res, out)

            print("\n---------->Validation Loss on Number of items {} : {}\n".format(num_items,loss))
            writer.add_scalar('MatNTM_Train_MatAssRec2/Validation_Loss_20items',loss,i+1)


            if loss < min_loss:
                min_loss = loss.detach().data
                Save()

            torch.cuda.empty_cache()


        # if (i+1)%1000 == 0:
        #     Save()
            
except ValueError as e:
	print("Value error... Get up: ", e)
	Save(False)