import numpy as np
import make_net as make
import random_provider as rp
import keras as k
import custom_loss as cl
import sys

    #####################
    #   initialize data #
    #####################

#load
raw=np.load("./data/spir_raw.npy")
aff=np.load("./data/spir_aff.npy")


#testing data
conf_raw=np.zeros((1,1,16,128,128))
conf_raw[0,0]=raw[100:116,100:228,100:228]
conf_raw=np.einsum("bczxy->bzxyc",conf_raw)

conf_aff=np.zeros((1,3,16,128,128))
conf_aff[0]=aff[:,100:116,100:228,100:228]
conf_aff=np.einsum("bczxy->bzxyc",conf_aff)

    #################
    #   setup model #
    #################

model=make.make()

adam= k.optimizers.Adam(lr=.000025)

WCE=cl.loss()

WCE.set_weight(.05)


#MULTI GPU STUFF!!!!
if (len(sys.argv)>1):
    GPU_N = sys.argv[1]
    if(GPU_N>1):
        print("running on %i GPUs..." %GPU_N)
        model=k.utils.multi_gpu_model(model,gpus=GPU_N)
    else:
        print("running in single GPU mode...")
else:
    print("no GPU num argument, running in single GPU mode... ")

model.compile(loss=WCE.weighted_cross,optimizer=adam,metrics=['accuracy'])




    #################
    #   run model   #
    #################

#run until terminated
a=1
iteartions=0
while(a==1):
    #grab random slices
    raw_in,aff_in=rp.random_provider_affin((16,128,128),raw,aff)
    model.fit(raw_in,aff_in,epochs=1)


    
    
    

    iteartions+=1
