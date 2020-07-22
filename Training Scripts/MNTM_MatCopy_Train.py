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



def MatrixCopyDataGen(batch_size = 32, item_size = [4,5], timesteps = 2, device = None):
    
    assert item_size[0] == item_size[1] - 1
    
    pre_content = torch.rand([batch_size, item_size[0]+1, item_size[1]*timesteps])
    
    pre_content[:,:-1,:][pre_content[:,:-1,:] > 0.5] = 1.0
    pre_content[:,-1,:] = 0.0
    pre_content[pre_content != 1] = 0.0
        
    limiter = torch.zeros([batch_size, item_size[0]+1, item_size[1]])
    limiter[:,-1,:] = 1.0
    #response_sheet = torch.zeros_like(pre_content)
    
    
    question = torch.cat([pre_content, limiter], dim = 2)
    
    return question.to(device), pre_content.to(device)









MNTM = MatrixNTM.MatNTM(batch_size=32, input_dims=[5,5], hidden_dims=[15,15], num_layers=2, output_dims=[5,5], num_RH=1, num_WH=1, NumSlots=140, MemorySlot_dims=[6,6], shift_range=1, device = 'cuda:0')
MNTM = MNTM.to('cuda:0')
print('Training on GPU .... ')
print("Training started for MEM_DIMS: {}".format([140,6,6]))                                                  #Change it on next line too!!       
print("Matrix Copy Task : No Beta Amplification : 140,6,6 : RMSprop : 1e-4 : 0.9 Momentum : 0.0 Weight Decay: 5,5 : 15,15   ")


#data = torch.load("Data1to20till50000.pth")

writer = SummaryWriter()





#nDNTM.CompleteReset()




loss_func = torch.nn.BCELoss()
optim = torch.optim.RMSprop(params = MNTM.parameters(), lr = 1e-4, momentum = 0.9)
#optim.load_state_dict(optim_state_dict)




#Backward Hooks for gradient clipping
for p in MNTM.parameters():
    p.register_hook(lambda grad: torch.clamp(grad, -20, 20)) 
print("All gradients will be clipped between {} ".format((-20,20)))





def Save(optim_dict_too = True):
	torch.save(MNTM, 'SavedModels/MatNTM_MatCopy1/INP5_5_HID30_30_MD50_6_6_Adam4CE_Copy50000/MNTM.pth')
	if optim_dict_too:
		torch.save(optim.state_dict(), 'SavedModels/MatNTM_MatCopy1/INP5_5_HID30_30_MD50_6_6_Adam4CE_Copy50000/optim_state_dict.pth')





num_epochs = 50000



try:

    for i in range(num_epochs):

        num_t =  int( torch.randint(low=1, high=31, size=[]) )
        
        inp, out = MatrixCopyDataGen(32,[4,5],num_t,'cuda:0')
        
        response_sheet = torch.zeros_like(out)

        print("Epoch: ",i)

        print("--->Sequence Length: ",int(num_t))

        MNTM.HiddenReset()

        MNTM.zero_grad()



        #Feeding the sequence
        for t in range(1,num_t+2):
            _ = MNTM(inp[:,:,(t-1)*5:t*5])

        #Taking Output from controller now, for backprop
        output = []

        for t in range(1,num_t+1):
            MNTM_out = MNTM(response_sheet[:,:,(t-1)*5:t*5])
            output.append(MNTM_out)


        res = torch.cat(output, dim=2)

        loss = loss_func(res, out)

        writer.add_scalar('MatNTM_Train_MatCopy2GPU/INP5_5_HID10_10_MD50_6_6_CE_Copy50000_Adam4',loss,i)

        print("----->Loss: {}".format(loss))


        loss.backward()

        #Gradients clipped...
        #torch.nn.utils.clip_grad_value_(MNTM.parameters(), clip_value=10)

        optim.step()
        
        torch.cuda.empty_cache()


        #Regular Check
        if (i+1)%100 == 0:

            # num_t = torch.randint(low=70,high=101,size=[])
            num_t = 60

            inp, out = MatrixCopyDataGen(32,[4,5],num_t, 'cuda:0')
            
            response_sheet = torch.zeros_like(out)

            print("Epoch: ",i)

            print("--->Sequence Length: ",int(num_t))

            MNTM.HiddenReset()

            MNTM.zero_grad()



            #Feeding the sequence
            for t in range(1,num_t+2):
                _ = MNTM(inp[:,:,(t-1)*5:t*5])

            #Taking Output from controller now, for backprop
            output = []

            for t in range(1,num_t+1):
                MNTM_out = MNTM(response_sheet[:,:,(t-1)*5:t*5])
                output.append(MNTM_out)


            res = torch.cat(output, dim=2)

            loss = loss_func(res, out)


            print("----->Loss: {}".format(loss))
            print("\n---------->Validation Loss on Sequence Length {} : {}\n".format(num_t,loss))
            writer.add_scalar('MatNTM_Train_MatCopy2GPU/Validation_Loss_60',loss,i+1)


            torch.cuda.empty_cache()

        if (i+1)%1000 == 0:
            Save()
	    

except ValueError as e:
	print("Value error... Get up: ", e)
	Save(False)





writer.close() 

# torch.save(nDNTM, 'SavedModels/NTM/RUN5_MD_120_20_CE_GNorm50_Copy50000/nDNTM.pth')
# torch.save(optim.state_dict(), 'SavedModels/NTM/RUN5_MD_120_20_CE_GNorm50_Copy50000/optim_state_dict.pth')