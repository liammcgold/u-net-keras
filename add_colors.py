import numpy as np
import tifffile as tif

def add_colors(gt):
    uniques=np.unique(gt)

    dict={}

    for num in uniques:
        rand_r=np.random.rand()
        rand_g=np.random.rand()
        rand_b=np.random.rand()
        dict[num]=[rand_r,rand_g,rand_b]

    new_image=np.zeros((np.shape(gt)[0],np.shape(gt)[1],np.shape(gt)[2],3))

    for i in range(0,np.shape(gt)[0]):
        for j in range(0,np.shape(gt)[1]):
            for k in range(0,np.shape(gt)[2]):
                new_image[i,j,k]=dict[gt[i,j,k]]

    return new_image





