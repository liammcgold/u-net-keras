import numpy as np


def random_provider_affin(shape,raw,affin):
    x=int(np.random.random()*(np.shape(raw)[0]-shape[0]))
    y=int(np.random.random()*(np.shape(raw)[1]-shape[1]))
    z=int(np.random.random()*(np.shape(raw)[2]-shape[2]))
    raw_out=np.zeros((1,1,shape[0],shape[1],shape[2]))
    aff_out=np.zeros((1,3,shape[0],shape[1],shape[2]))
    raw_out[0]=raw[x:x+shape[0],y:y+shape[1],z:z+shape[2]]
    aff_out[0]=affin[:,x:x+shape[0],y:y+shape[1],z:z+shape[2]]
    raw_out=np.einsum("bczxy->bzxyc",raw_out)
    aff_out = np.einsum("bczxy->bzxyc", aff_out)


    return raw_out,aff_out


def random_provider_flat(shape,raw,flat):
    x=int(np.random.random()*(np.shape(raw)[0]-shape[0]))
    y=int(np.random.random()*(np.shape(raw)[1]-shape[1]))
    z=int(np.random.random()*(np.shape(raw)[2]-shape[2]))
    return raw[x:x+shape[0],y:y+shape[1],z:z+shape[2]], flat[x:x+shape[0],y:y+shape[1],z:z+shape[2]]


def random_provider_raw(shape,raw):
    x=int(np.random.random()*(np.shape(raw)[0]-shape[0]))
    y=int(np.random.random()*(np.shape(raw)[1]-shape[1]))
    z=int(np.random.random()*(np.shape(raw)[2]-shape[2]))
    return raw[x:x+shape[0],y:y+shape[1],z:z+shape[2]]