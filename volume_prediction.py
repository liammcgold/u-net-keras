import keras
import numpy as np
import assembler
import u_net_MG


model=u_net_MG.make()

itteration=1699452

model.load_weights("saved_models_MG/model%i"%itteration)

def predict(raw_block):


    #temp=np.zeros((np.shape(raw_block)[aff_graph0]*2,np.shape(raw_block)[2]*2,np.shape(raw_block)[2]*2))

    #temp[int(np.shape(raw_block)[aff_graph0]/2):int(np.shape(raw_block)[aff_graph0]/2)+np.shape(raw_block)[aff_graph0],int(np.shape(raw_block)[1]/2):int(np.shape(raw_block)[1]/2)+np.shape(raw_block)[1],int(np.shape(raw_block)[2]/2):int(np.shape(raw_block)[2]/2)+np.shape(raw_block)[2]]=raw_block

    #temp=np.reshape(temp,(1,np.shape(temp)[aff_graph0],np.shape(temp)[1],np.shape(temp)[2],1))

    raw_block = np.reshape(raw_block, (1, 16, 128, 128, 1))

    aff = model.predict(raw_block)

    aff = np.einsum("bzxyc->bczxy", aff)

    aff = aff[0]

    return aff


raw=np.load("data/spir_raw.npy")


raw=raw[0:32,:,:]


overlap_out_of_16=12

blending_factor=1

asmblr=assembler.assembler(raw,overlap_out_of_16/16,(16,128,128), predict,blending_factor)


aff=asmblr.process()

np.save("data/spir_bin2_aff%i"%itteration,aff)



