import keras as k
import path_join as pj



def make():

    raw_input=k.layers.Input((16,128,128,1))

    #########################
    #   Large Kernel Path   #
    #########################

    lks = 9

    # cl0
    cl0 = k.layers.Conv3D(8, (1, lks, lks), padding="same", activation="relu")(raw_input)
    cl01 = k.layers.Conv3D(8, (lks, lks, lks), padding="same", activation="relu")(cl0)
    cl0mp = k.layers.Conv3D(8, (lks, lks, lks), padding="same", activation="relu")(raw_input)
    cl0m = k.layers.add([cl0mp, cl01])
    cl0m = k.layers.BatchNormalization()(cl0m)

    # dl0
    dl0 = k.layers.MaxPool3D([1, 2, 2])(cl0m)

    # cl1
    cl1 = k.layers.Conv3D(8, (1, lks, lks), padding="same", activation="relu")(dl0)
    cl11 = k.layers.Conv3D(8, (lks, lks, lks), padding="same", activation="relu")(cl1)
    cl1mp = k.layers.Conv3D(8, (lks, lks, lks), padding="same", activation="relu")(dl0)
    cl1m = k.layers.add([cl1mp, cl11])
    cl1m = k.layers.BatchNormalization()(cl1m)

    # ml0
    ul0 = k.layers.UpSampling3D([1, 2, 2])(cl1m)
    ml0p = k.layers.LeakyReLU()(ul0)
    cl0mp0 = k.layers.LeakyReLU()(cl0m)
    ml0 = k.layers.add([ml0p, cl0mp0])

    # mcl0
    mcl0 = k.layers.Conv3D(3, (1, lks, lks), padding="same", activation="relu")(ml0)
    mcl01 = k.layers.Conv3D(3, (lks, lks, lks), padding="same", activation="relu")(mcl0)
    mcl0mp = k.layers.Conv3D(3, (lks, lks, lks), padding="same", activation="relu")(ml0)
    mcl0m = k.layers.add([mcl0mp, mcl01])
    mcl0m = k.layers.BatchNormalization()(mcl0m)

    o1 = k.layers.LeakyReLU()(mcl0m)

    #################
    #   Join Paths  #
    #################

    out=k.layers.LeakyReLU()(o1)
    out=k.layers.BatchNormalization()(out)

    model=k.models.Model(inputs=raw_input,outputs=out)

    print(model.summary())

    k.utils.plot_model(model,"model_LK.png",show_shapes=True)

    return model

