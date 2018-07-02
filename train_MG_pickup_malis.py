#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import u_net_MG as make
import random_provider as rp
import keras as k
import custom_loss as cl
import sys
import tifffile
import sklearn
import time
import malis_loss


    ##########################
    #   select model to load #
    ##########################

num = 1699452

    #####################
    #   initialize data #
    #####################

#load
raw=np.load("./data/spir_raw.npy")
aff=np.load("./data/spir_aff_2.npy")
gt=np.load("./data/spir_gt.npy")




loss_info=np.zeros((np.shape(aff)[0]+1,np.shape(aff)[1],np.shape(aff)[2],np.shape(aff)[3]))

loss_info[0:3]=aff
loss_info[3]=gt



#testing data
conf_raw=np.zeros((1,1,16,128,128))
conf_raw[0,0]=raw[200:216,200:328,200:328]
conf_raw=np.einsum("bczxy->bzxyc",conf_raw)

conf_aff=np.zeros((1,3,16,128,128))
conf_aff[0]=aff[:,200:216,200:328,200:328]
conf_aff=np.einsum("bczxy->bzxyc",conf_aff)


conf_gt=gt[200:216,200:328,200:328]



#weights for loss
#weights=np.zeros((3,2))
#
#
# weights[0]=sklearn.utils.class_weight.compute_class_weight('balanced',
#                                                         np.unique(aff[0]),
#                                                         aff[0].flatten())
# weights[1]=sklearn.utils.class_weight.compute_class_weight('balanced',
#                                                         np.unique(aff[1]),
#                                                         aff[1].flatten())
# weights[2]=sklearn.utils.class_weight.compute_class_weight('balanced',
#                                                         np.unique(aff[2]),
#                                                        aff[2].flatten())
#
# weights[0]=[2.6960856,0.61383891]
# weights[1]=[4.05724285,0.57027915]
# weights[2]=[4.09752934, 0.56949214]
#
# weights=np.asarray(weights)
# WCE=cl.loss()
#
# WCE.set_weight(weights)



    #################
    #   setup model #
    #################

model=make.make()

model.load_weights("saved_models_MG/model%i"%num)


adam= k.optimizers.Adam(lr=.0000025)

loss=malis_loss.mal((16,128,128))






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

model.compile(loss=loss.malis,optimizer=adam,metrics=['accuracy'])




    #################
    #   run model   #
    #################

k.utils.plot_model(model, "model_variant_a.png", show_shapes=True)

#run until terminated
a=1
i=num
pred_old=[1]
try:
    while(a==1):
        time_str=time.time()


            #########################
            #   grab random slices  #
            #########################

        raw_in,loss_info_in=rp.random_provider_loss_info((16,128,128),raw,loss_info)

            #############
            #   train   #
            #############

        model.fit(raw_in,loss_info_in,epochs=1)



            #############################
            #  test and export images   #
            #############################

        if (i % 10 == 0):

            pred=model.predict(conf_raw)

            if (np.shape(pred_old)[0]>1):
                if(i>20):
                    if (np.array_equal(pred_old, pred)):
                        print("CONVERGED TO BAD RESULT")
                        break

            for j in range(0, 16):
                tifffile.imsave("tiffs/pred0/0predicted_affins%i" % j,
                                np.asarray(pred, dtype=np.float32)[0, j,: , :, 0])
                tifffile.imsave("tiffs/pred1/1predicted_affins%i" % j,
                                np.asarray(pred, dtype=np.float32)[0, j,: , :, 1])
                tifffile.imsave("tiffs/pred2/2predicted_affins%i" % j,
                                np.asarray(pred, dtype=np.float32)[0, j,: , :, 2])
                tifffile.imsave("tiffs/act0/0actual_affins%i" % j,
                                np.asarray(conf_aff, dtype=np.float32)[0, j,:, :, 0])
                tifffile.imsave("tiffs/act1/1actual_affins%i" % j,
                                np.asarray(conf_aff, dtype=np.float32)[0, j,:, :, 1])
                tifffile.imsave("tiffs/act2/2actual_affins%i" % j,
                                np.asarray(conf_aff, dtype=np.float32)[0, j,:, :, 2])
                tifffile.imsave("tiffs/raw/raw%i" % j,
                                np.asarray(conf_raw, dtype=np.float32)[0, j, :, :, 0])
                tifffile.imsave("tiffs/gt/gt%i" % j,
                                np.asarray(conf_gt, dtype=np.float32)[j, :, :])


            np.save("data_out/prediction",pred)

            pred_old=pred

            ###########################################
            # Save Based on Iterations to Save Memory #
            ###########################################

        if (i < 100):
            if (i % 10 == 0):
                model.save("./saved_models_MG/model%i"%i)
        if (i >= 100 and i < 1000):
            if (i % 100 == 0):
                model.save("./saved_models_MG/model%i" % i)
        if (i >= 1000 and i < 1000):
            if (i % 1000 == 0):
                model.save("./saved_models_MG/model%i" % i)
        if (i >= 10000 and i < 100000):
            if (i % 5000 == 0):
                model.save("./saved_models_MG/model%i" % i)
        if (i >= 100000):
            if (i % 20000 == 0):
                model.save("./saved_models_MG/model%i" % i)

        print("Iteration: %i"%i)
        i+=1
        time_stop=time.time()
        t=time_stop-time_str
        print(t)



except KeyboardInterrupt:
    print("SAVING MODEL EARLY")
    model.save("./saved_models_MG/model%i" % i)