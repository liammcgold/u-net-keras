import keras as k


def make():
    raw_input = k.layers.Input((16, 128, 128, 1))

    #########################
    #   Small Kernel Path   #
    #########################

    sks = 3

    # c0
    c0 = k.layers.Conv3D(8, (1, sks, sks), padding="same", activation="relu")(raw_input)
    c0mp = k.layers.Conv3D(8, (sks, sks, sks), padding="same", activation="relu")(raw_input)
    c0m = k.layers.add([c0mp, c0])
    c0m = k.layers.BatchNormalization()(c0m)

    # d0
    d0 = k.layers.MaxPool3D([1, 2, 2])(c0m)

    # c1
    c1 = k.layers.Conv3D(32, (1, sks, sks), padding="same", activation="relu")(d0)
    c1mp = k.layers.Conv3D(32, (sks, sks, sks), padding="same", activation="relu")(d0)
    c1m = k.layers.add([c1mp, c1])
    c1m = k.layers.BatchNormalization()(c1m)

    # d1
    d1 = k.layers.MaxPool3D([1, 2, 2])(c1m)

    # c2
    c2 = k.layers.Conv3D(64, (1, sks, sks), padding="same", activation="relu")(d1)
    c2mp = k.layers.Conv3D(64, (sks, sks, sks), padding="same", activation="relu")(d1)
    c2m = k.layers.add([c2mp, c2])
    c2m = k.layers.BatchNormalization()(c2m)

    # d2
    d2 = k.layers.MaxPool3D([1, 2, 2])(c2m)

    # c3
    c3 = k.layers.Conv3D(64, (1, sks, sks), padding="same", activation="relu")(d2)
    c3mp = k.layers.LeakyReLU()(d2)
    c3m = k.layers.add([c3mp, c3])
    c3m = k.layers.BatchNormalization()(c3m)

    # m0
    u0 = k.layers.UpSampling3D([1, 2, 2])(c3m)
    m0p = k.layers.LeakyReLU()(u0)
    c2mp0 = k.layers.LeakyReLU()(c2m)
    m0 = k.layers.add([m0p, c2mp0])

    # mc0
    mc0 = k.layers.Conv3D(32, (1, sks, sks), padding="same", activation="relu")(m0)
    mc0mp = k.layers.Conv3D(32, (sks, sks, sks), padding="same", activation="relu")(m0)
    mc0m = k.layers.add([mc0mp, mc0])
    mc0m = k.layers.BatchNormalization()(mc0m)

    # m1
    u1 = k.layers.UpSampling3D([1, 2, 2])(mc0m)
    m1p = k.layers.LeakyReLU()(u1)
    c3mp0 = k.layers.LeakyReLU()(c1m)
    m1 = k.layers.add([m1p, c3mp0])

    # mc1
    mc1 = k.layers.Conv3D(8, (1, sks, sks), padding="same", activation="relu")(m1)
    mc1mp = k.layers.Conv3D(8, (sks, sks, sks), padding="same", activation="relu")(m1)
    mc1m = k.layers.add([mc1mp, mc1])
    mc1m = k.layers.BatchNormalization()(mc1m)

    # m2
    u2 = k.layers.UpSampling3D([1, 2, 2])(mc1m)
    m2p = k.layers.LeakyReLU()(u2)
    c0mp0 = k.layers.LeakyReLU()(c0m)
    m2 = k.layers.add([m2p, c0mp0])

    # mc2
    mc2 = k.layers.Conv3D(3, (1, sks, sks), padding="same", activation="relu")(m2)
    mc2mp = k.layers.Conv3D(3, (sks, sks, sks), padding="same", activation="relu")(m2)
    mc2m = k.layers.add([mc2mp, mc2])
    mc2m = k.layers.BatchNormalization()(mc2m)

    o0 = k.layers.LeakyReLU()(mc2m)

    #########################
    #   Large Kernel Path   #
    #########################

    lks = 9

    # cl0
    cl0 = k.layers.Conv3D(8, (1, lks, lks), padding="same", activation="relu")(raw_input)
    cl0mp = k.layers.Conv3D(8, (lks, lks, lks), padding="same", activation="relu")(raw_input)
    cl0m = k.layers.add([cl0mp, cl0])
    cl0m = k.layers.BatchNormalization()(cl0m)

    # dl0
    dl0 = k.layers.MaxPool3D([1, 2, 2])(cl0m)

    # cl1
    cl1 = k.layers.Conv3D(8, (1, lks, lks), padding="same", activation="relu")(dl0)
    cl1mp = k.layers.Conv3D(8, (lks, lks, lks), padding="same", activation="relu")(dl0)
    cl1m = k.layers.add([cl1mp, cl1])
    cl1m = k.layers.BatchNormalization()(cl1m)

    # ml0
    ul0 = k.layers.UpSampling3D([1, 2, 2])(cl1m)
    ml0p = k.layers.LeakyReLU()(ul0)
    cl0mp0 = k.layers.LeakyReLU()(cl0m)
    ml0 = k.layers.add([ml0p, cl0mp0])

    # mcl0
    mcl0 = k.layers.Conv3D(3, (1, lks, lks), padding="same", activation="relu")(ml0)
    mcl0mp = k.layers.Conv3D(3, (lks, lks, lks), padding="same", activation="relu")(ml0)
    mcl0m = k.layers.add([mcl0mp, mcl0])
    mcl0m = k.layers.BatchNormalization()(mcl0m)

    o1 = k.layers.LeakyReLU()(mcl0m)

    #################
    #   Join Paths  #
    #################

    join = pj.path_join(o0, o1, 3)
    out = k.layers.LeakyReLU()(join)
    out = k.layers.BatchNormalization(center=.5, )(out)
    out = k.layers.Conv3D(3, (lks, lks, lks), padding="same", activation="sigmoid")(out)

    model = k.models.Model(inputs=raw_input, outputs=out)

    print(model.summary())

    k.utils.plot_model(model, "model_MG.png", show_shapes=True)

    return model

