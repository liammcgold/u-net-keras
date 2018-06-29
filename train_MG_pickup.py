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

    ##########################
    #   select model to load #
    ##########################

num = 1020000

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


#weights for loss
weights=np.zeros((3,2))

# weights[0]=sklearn.utils.class_weight.compute_class_weight('balanced',
#                                                         np.unique(aff[0]),
#                                                         aff[0].flatten())
# weights[1]=sklearn.utils.class_weight.compute_class_weight('balanced',
#                                                         np.unique(aff[1]),
#                                                         aff[1].flatten())
# weights[2]=sklearn.utils.class_weight.compute_class_weight('balanced',
#                                                         np.unique(aff[2]),
#                                                         aff[2].flatten())

#print(weights)


#hard coded for speedup since its always the same
weights[0]=[ 4.77092571,  0.55853532]
weights[1]=[25.34738088,  0.51006142]
weights[2]=[27.62890733,  0.50921526]



    #################
    #   setup model #
    #################

model=make.make()

model.load_weights("saved_models_MG/model%i"%num)


adam= k.optimizers.Adam(lr=.0000025)

WCE=cl.loss()

#added *2 to see if it improved white to black ratio (too much black)
WCE.set_weight(weights)


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

k.utils.plot_model(model, "model_variant_a.png", show_shapes=True)

#run until terminated
a=1
i=num
pred_old=[1]
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
