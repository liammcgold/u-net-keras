#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import u_net_SK as make
import random_provider as rp
import keras as k
import custom_loss as cl
import sys
import tifffile

    #####################
    #   initialize data #
    #####################

#load
raw=np.load("./data/spir_raw.npy")
aff=np.load("./data/spir_aff.npy")


#testing data
conf_raw=np.zeros((1,1,16,128,128))
conf_raw[0,0]=raw[200:216,200:328,200:328]
conf_raw=np.einsum("bczxy->bzxyc",conf_raw)

conf_aff=np.zeros((1,3,16,128,128))
conf_aff[0]=aff[:,200:216,200:328,200:328]
conf_aff=np.einsum("bczxy->bzxyc",conf_aff)

    #################
    #   setup model #
    #################

model=make.make()

adam= k.optimizers.Adam(lr=.0000025)

WCE=cl.loss()

WCE.set_weight(.5)


#MULTI GPU STUFF!!!!
if (len(sys.argv)>1):
    GPU_N = sys.argv[1]
    if(int(GPU_N)>1):
        print("\nRunning on %i GPUs...\n" %GPU_N)
        model=k.utils.multi_gpu_model(model,gpus=GPU_N)
    else:
        print("\nRunning in single GPU mode...\n")
else:
    print("\nNo GPU num argument, running in single GPU mode... \n")

model.compile(loss=WCE.weighted_cross,optimizer=adam,metrics=['accuracy'])




    #################
    #   run model   #
    #################



#run until terminated
a=1
i=0
while(a==1):

        #########################
        #   grab random slices  #
        #########################

    raw_in,aff_in=rp.random_provider_affin((16,128,128),raw,aff)

        #############
        #   train   #
        #############

    model.fit(raw_in,aff_in,epochs=1)

        #############################
        #  test and export images   #
        #############################

    if (i % 10 == 0):

        pred=model.predict(conf_raw)

        if(i>20):
            if (np.array_equal(pred_old, pred)):
                print("CONVERGED TO BAD RESULT")
                break

        for j in range(0, 16):
            tifffile.imsave("tiffs/pred/predicted_affins%i" % j,
                            np.asarray(pred, dtype=np.float32)[0, j,: , :, 0])
            tifffile.imsave("tiffs/act/actual_affins%i" % j,
                            np.asarray(conf_aff, dtype=np.float32)[0, j,:, :, 0])
            tifffile.imsave("tiffs/raw/raw%i" % j,
                            np.asarray(conf_raw, dtype=np.float32)[0, j, :, :, 0])


        pred_old=pred

        ###########################################
        # Save Based on Iterations to Save Memory #
        ###########################################

    if (i < 100):
        if (i % 10 == 0):
            model.save("./saved_models_SK/model%i"%i)
    if (i >= 100 and i < 1000):
        if (i % 100 == 0):
            model.save("./saved_models_SK/model%i" % i)
    if (i >= 1000 and i < 1000):
        if (i % 1000 == 0):
            model.save("./saved_models_SK/model%i" % i)
    if (i >= 10000 and i < 100000):
        if (i % 5000 == 0):
            model.save("./saved_models_SK/model%i" % i)
    if (i >= 100000):
        if (i % 20000 == 0):
            model.save("./saved_models_SK/model%i" % i)

    print("Iteration: %i"%i)
    i+=1
