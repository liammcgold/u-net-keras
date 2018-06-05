import keras as k


def path_join(input_1,input_2,filters):

    s0=k.layers.add([input_1,input_2])

    r0=k.layers.LeakyReLU()(input_1)
    r1=k.layers.LeakyReLU()(s0)
    r2=k.layers.LeakyReLU()(input_2)

    s1=k.layers.add([r0,r1,r2])

    c0=k.layers.Conv3D(filters,(8,8,8),padding="same",activation="relu")(r0)
    c1=k.layers.Conv3D(filters,(8,8,8),padding="same",activation="relu")(s1)
    c2=k.layers.Conv3D(filters,(8,8,8),padding="same",activation="relu")(r2)

    s2=k.layers.add([c0,c1,c2])

    return s2