import keras as k
import path_join as pj



def make():

    raw_input=k.layers.Input((16,128,128,1))


    #########################
    #   Small Kernel Path   #
    #########################

    #c0
    c0=k.layers.Conv3D(8,(1,3,3),padding="same",activation="relu")(raw_input)
    c00=k.layers.Conv3D(8,(3,3,3),padding="same",activation="relu")(c0)
    c01=k.layers.Conv3D(8,(3,3,3),padding="same",activation="relu")(c00)
    c0mp=k.layers.Conv3D(8,(3,3,3),padding="same",activation="relu")(raw_input)
    c0m=k.layers.add([c0mp,c01])

    #d0
    d0=k.layers.MaxPool3D([1,2,2])(c0m)

    #c1
    c1 = k.layers.Conv3D(32, (3,3,3), padding="same", activation="relu")(d0)
    c10 = k.layers.Conv3D(32, (3,3,3), padding="same", activation="relu")(c1)
    c11 = k.layers.Conv3D(32, (3,3,3), padding="same", activation="relu")(c10)
    c1mp=k.layers.Conv3D(32, (3,3,3), padding="same", activation="relu")(d0)
    c1m = k.layers.add([c1mp, c11])

    #d1
    d1=k.layers.MaxPool3D([1,2,2])(c1m)

    #c2
    c2 = k.layers.Conv3D(64, (3,3,3), padding="same", activation="relu")(d1)
    c20 = k.layers.Conv3D(64, (3,3,3), padding="same", activation="relu")(c2)
    c21 = k.layers.Conv3D(64, (3,3,3), padding="same", activation="relu")(c20)
    c2mp = k.layers.Conv3D(64, (3,3,3), padding="same", activation="relu")(d1)
    c2m = k.layers.add([c2mp, c21])

    #d2
    d2=k.layers.MaxPool3D([1,2,2])(c2m)

    #c3
    c3 = k.layers.Conv3D(64, (3,3,3), padding="same", activation="relu")(d2)
    c30 = k.layers.Conv3D(64, (3,3,3), padding="same", activation="relu")(c3)
    c31 = k.layers.Conv3D(64, (3,3,3), padding="same", activation="relu")(c30)
    c3mp = k.layers.LeakyReLU()(d2)
    c3m = k.layers.add([c3mp, c31])


    #m0
    u0=k.layers.UpSampling3D([1,2,2])(c3m)
    m0p=k.layers.LeakyReLU()(u0)
    c2mp0=k.layers.LeakyReLU()(c2m)
    m0=k.layers.add([m0p,c2mp0])


    #mc0
    mc0 = k.layers.Conv3D(32, (3,3,3), padding="same", activation="relu")(m0)
    mc00 = k.layers.Conv3D(32, (3,3,3), padding="same", activation="relu")(mc0)
    mc01 = k.layers.Conv3D(32, (3,3,3), padding="same", activation="relu")(mc00)
    mc0mp = k.layers.Conv3D(32, (3,3,3), padding="same", activation="relu")(m0)
    mc0m = k.layers.add([mc0mp, mc01])

    # m1
    u1 = k.layers.UpSampling3D([1, 2, 2])(mc0m)
    m1p = k.layers.LeakyReLU()(u1)
    c3mp0 = k.layers.LeakyReLU()(c1m)
    m1 = k.layers.add([m1p, c3mp0])

    # mc1
    mc1 = k.layers.Conv3D(8, (3,3,3), padding="same", activation="relu")(m1)
    mc10 = k.layers.Conv3D(8, (3,3,3), padding="same", activation="relu")(mc1)
    mc11 = k.layers.Conv3D(8, (3,3,3), padding="same", activation="relu")(mc10)
    mc1mp = k.layers.Conv3D(8, (3,3,3), padding="same", activation="relu")(m1)
    mc1m = k.layers.add([mc1mp, mc11])

    # m2
    u2 = k.layers.UpSampling3D([1, 2, 2])(mc1m)
    m2p = k.layers.LeakyReLU()(u2)
    c0mp0 = k.layers.LeakyReLU()(c0m)
    m2 = k.layers.add([m2p, c0mp0])

    # mc2
    mc2 = k.layers.Conv3D(3, (3,3,3), padding="same", activation="relu")(m2)
    mc20 = k.layers.Conv3D(3, (3,3,3), padding="same", activation="relu")(mc2)
    mc21 = k.layers.Conv3D(3, (3,3,3), padding="same", activation="relu")(mc20)
    mc2mp = k.layers.Conv3D(3, (3,3,3), padding="same", activation="relu")(m2)
    mc2m = k.layers.add([mc2mp, mc21])

    o0 = k.layers.LeakyReLU()(mc2m)

    #########################
    #   Large Kernel Path   #
    #########################

    #cl0
    cl0=k.layers.Conv3D(8,(16,16,16),padding="same",activation="relu")(raw_input)
    cl00=k.layers.Conv3D(8,(16,16,16),padding="same",activation="relu")(cl0)
    cl01 = k.layers.Conv3D(8, (16, 16, 16), padding="same", activation="relu")(cl00)
    cl0mp = k.layers.Conv3D(8, (16, 16, 16), padding="same", activation="relu")(raw_input)
    cl0m= k.layers.add([cl0mp,cl01])

    #dl0
    dl0=k.layers.MaxPool3D([1,2,2])(cl0m)

    # cl1
    cl1 = k.layers.Conv3D(8, (16, 16, 16), padding="same", activation="relu")(dl0)
    cl10 = k.layers.Conv3D(32, (16, 16, 16), padding="same", activation="relu")(cl1)
    cl11 = k.layers.Conv3D(8, (16, 16, 16), padding="same", activation="relu")(cl10)
    cl1mp = k.layers.Conv3D(8, (16, 16, 16), padding="same", activation="relu")(dl0)
    cl1m = k.layers.add([cl1mp, cl11])

    #ml0
    ul0=k.layers.UpSampling3D([1,2,2])(cl1m)
    ml0p=k.layers.LeakyReLU()(ul0)
    cl0mp0=k.layers.LeakyReLU()(cl0m)
    ml0=k.layers.add([ml0p,cl0mp0])

    #mcl0
    mcl0 = k.layers.Conv3D(3, (16, 16, 16), padding="same", activation="relu")(ml0)
    mcl00 = k.layers.Conv3D(3, (16, 16, 16), padding="same", activation="relu")(mcl0)
    mcl01 = k.layers.Conv3D(3, (16, 16, 16), padding="same", activation="relu")(mcl00)
    mcl0mp = k.layers.Conv3D(3, (16, 16, 16), padding="same", activation="relu")(ml0)
    mcl0m = k.layers.add([mcl0mp, mcl01])

    o1=k.layers.LeakyReLU()(mcl0m)

    #################
    #   Join Paths  #
    #################

    join=pj.path_join(o0,o1,3)
    out=k.layers.LeakyReLU()(join)

    model=k.models.Model(inputs=raw_input,outputs=out)

    print(model.summary())

    k.utils.plot_model(model,"model_variant.png",show_shapes=True)

    return model

make()