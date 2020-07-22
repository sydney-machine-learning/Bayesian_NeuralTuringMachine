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


MNTM = MatrixNTM.MatNTM(batch_size=16, input_dims=[6,5], hidden_dims=[15,15], num_layers=3, output_dims=[6,5], num_RH=1, num_WH=1, NumSlots=120, MemorySlot_dims=[6,6], shift_range=1, device='cuda:0')

MNTM = MNTM.to('cuda:0')
print('Training on GPU...')

print("Training started for MEM_DIMS: {}".format([120,6,6]))                                                  #Change it on next line too!!       

print("Matrix Repeat Copy Task : 120,6,6 : RMSprop : 1e-4 : 6,6 : 15,15 : 3 layers : Centered RMSprop : ReduceLROnPlateau")




writer = SummaryWriter()





#nDNTM.CompleteReset()



# def CrossEntropyLoss(true, res, eps = 1e-10):
#   	return -torch.mean(true*torch.log(res+eps) + (1-true)*torch.log(1-res+eps))


loss_func = torch.nn.BCELoss()
optim = torch.optim.RMSprop(params = MNTM.parameters(), lr = 1e-4, momentum = 0.9, centered = True)
#optim.load_state_dict(optim_state_dict)

lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode='min', factor=0.9, patience=150, verbose=True, min_lr=5e-6)



# Backward Hooks for gradient clipping
# for p in MNTM.parameters():
#     p.register_hook(lambda grad: torch.clamp(grad, -10, 10)) 
# print("All gradients will be clipped between {} ".format((-10,10)))





def RepeatCopy(batch_size = 32, inp_size = [6,5], seq_length = 4, num_repeats = 2, max_repeats = 30, device=None):
    
    #assert (inp_size[0] - 1)*inp_size[1] >= num_repeats, 'num_repeats too high.'
    
    sequence = torch.rand([batch_size, inp_size[0], inp_size[1]*seq_length])
    sequence[:,inp_size[1]-1:,:] = 0.0
    
    sequence[sequence>0.5] = 1.0
    sequence[sequence<=0.5] = 0.0
    
#     eof_lim = torch.zeros([batch_size,*inp_size])
#     eof_lim[:,-1,0] = 1.0
#     eof_lim[:,-1,-1] = 1.0
    
#     repeats = torch.zeros([batch_size, (inp_size[0]-1)*inp_size[1]])
#     repeats[:,(inp_size[0]-1)*inp_size[1] - num_repeats:] = 1.0
#     repeats = repeats.reshape([batch_size, inp_size[0]-1, inp_size[1]])
#     repeats = torch.cat([repeats, torch.zeros([batch_size,1, inp_size[1]])], dim = 1)
    
    #Assuming repeats \sim Unif(1,max_repeats)
    repeats = torch.zeros([batch_size, *inp_size])
    mean = (max_repeats - 1)/2
    std = (((max_repeats )**2 - 1)/12 )**0.5
    repeats[:,-1,1:-1] = ( num_repeats - mean ) / std
    
    op_start_delim = torch.zeros([batch_size,*inp_size])
    op_start_delim[:,-1,0] = 1.0
    op_start_delim[:,-1,-1] = 1.0
    
    
    output = sequence.repeat([1,1,num_repeats])
    output = torch.cat([output, op_start_delim], dim=2)
    
    sequence = torch.cat([ op_start_delim, sequence, op_start_delim, repeats], dim = 2)
    
    
    return sequence.to(device), output.to(device)




def Save(optim_dict_too = True):
	torch.save(MNTM, 'SavedModels/MatNTM_Train_MatRepCopy1Cuda/INP6_6_HID30_30_MD120_6_6_Adam4CE_RepCopy100000/MNTM.pth')
	if optim_dict_too:
		torch.save(optim.state_dict(), 'SavedModels/MatNTM_Train_MatRepCopy1Cuda/INP6_6_HID30_30_MD120_6_6_Adam4CE_RepCopy100000/optim_state_dict.pth')
		torch.save(lr_schedule.state_dict(), 'SavedModels/MatNTM_Train_MatRepCopy1Cuda/INP6_6_HID30_30_MD120_6_6_Adam4CE_RepCopy100000/lr_scheduler_state_dict.pth')


def Bit_Seq_Error(output , target, batch_size=16):
    bit_output = output.detach().clone()
    bit_output[bit_output >= 0.5] = 1.0
    bit_output[bit_output != 1.0] = 0.0
    bit_error = torch.sum((torch.abs(target - bit_output)) )/batch_size
    return bit_error



num_epochs = 100000









min_loss = 0.7

try:

    for i in range(num_epochs):


        seq_length = torch.randint(low=1, high=11, size=[])
        num_repeats = torch.randint(low=1, high=11, size=[])

        inp, out = RepeatCopy(batch_size=16, inp_size=[6,5], seq_length=seq_length, num_repeats=num_repeats, max_repeats = 10, device='cuda:0')
        #inp, out = MatrixCopyDataGen(32,[4,5],num_t)

        response_sheet = torch.zeros_like(out)

        print("Epoch: ",i)
        print("--> Current LR : {}\n".format(optim.param_groups[0]['lr']))
        print("--->Sequence length : {}".format(seq_length) )
        print("--->No. of Repetitions: {}".format(num_repeats) )

        MNTM.HiddenReset()

        MNTM.zero_grad()



        #Feeding the sequence
        #Input sequence is of length inp.shape[2]/6 

        num_sequences = int(inp.shape[2]/5)


        for t in range(1,num_sequences+1):
            _ = MNTM(inp[:,:,(t-1)*5:t*5])
                               #6 is the inp_size[1]                    
		#del inp

        #Taking Output from controller now, for backprop
        output = []
                #num_repeats + 1 (+1 for eof delimiter)
        for t in range(1, (num_repeats*seq_length +1)  +1):
        	MNTM_out = MNTM(response_sheet[:,:,(t-1)*5:t*5])
        	output.append(MNTM_out)

        #del response_sheet

        res = torch.cat(output, dim=2)


        loss = loss_func(res, out)                               #^ is the batch size.
        bit_seq_error = Bit_Seq_Error(output = res, target = out, batch_size= 16)


        #writer.add_scalar('MatNTM_Train_MatRepCopy4/INP6_6_HID30_30_MD120_6_6_Adam4CE_RepCopy100000',loss,i)

        print("----->Loss: {}".format(float(loss)))
        print("----->Bit Sequence Error : {}".format(float(bit_seq_error)))
        #del out

        writer.add_scalar('MatNTM_Train_MatRepCopy2GPU/Train Loss SL1-10 NR1-10',loss,i)
        writer.add_scalar('MatNTM_Train_MatRepCopy2GPU/Train Loss SL1-10 NR1-10 BitSeqError',bit_seq_error,i)


        loss.backward()




        #Gradients clipped...
        torch.nn.utils.clip_grad_value_(MNTM.parameters(), clip_value=20)

        optim.step()


        if (loss.detach().data<=0.05) and (seq_length >= 5) and (num_repeats>=8):
        	lr_schedule.step(bit_seq_error)



        #del res

        #del loss

        torch.cuda.empty_cache()




        if (i+1)%100 == 0:

            seq_length = 15#torch.randint(low=1, high=6, size=[])

            num_repeats = 5#torch.randint(low=1, high=16, size=[])

            inp, out = RepeatCopy(batch_size=16, inp_size=[6,5], seq_length=seq_length, num_repeats=num_repeats, max_repeats=10, device = 'cuda:0')
            #inp, out = MatrixCopyDataGen(32,[4,5],num_t)

            response_sheet = torch.zeros_like(out)

            print("\n ----> Validation Run <----")

            print("--->Sequence length : {}".format(seq_length) )
            print("--->No. of Repetitions: {}".format(num_repeats) )

            MNTM.HiddenReset()

            #MNTM.zero_grad()



            #Feeding the sequence
            #Input sequence is of length inp.shape[2]/6 

            num_sequences = int(inp.shape[2]/5)


            for t in range(1,num_sequences+1):
                _ = MNTM(inp[:,:,(t-1)*5:t*5])
                                           #6 is the inp_size[1]                    


            #del inp
            #Taking Output from controller now, for backprop
            output = []
                            #num_repeats + 1 (+1 for eof delimiter)
            for t in range(1, (num_repeats*seq_length +1)  +1):
                MNTM_out = MNTM(response_sheet[:,:,(t-1)*5:t*5])
                output.append(MNTM_out)    

            #del response_sheet

            res = torch.cat(output, dim=2)

           
            loss = loss_func(res, out)

            #del out

            writer.add_scalar('MatNTM_Train_MatRepCopy2GPU/Validation_loss_SL15_NR5',loss,i)

            print("-------------->Validation Loss: {}".format(loss))
            print("-------------->Validation Bit Seq Error: {}".format(Bit_Seq_Error(output = res, target = out, batch_size= 16)))
            #del res

            #del loss
			
            torch.cuda.empty_cache()

            if loss.detach().data < min_loss:
            	min_loss = loss.detach().data
            	print("Minimum Loss on Validation Set .... Saving current model and optimizer parameters ...")
            	Save()

        if (i+1)%1000 == 0:
            Save()
            
        
except ValueError as e:
	print("Value error... Get up: ", e)
	Save(False)    

