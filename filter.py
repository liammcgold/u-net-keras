import numpy as np
import time

def filter(seg,thresh):

    new_seg=seg

    uniques=np.unique(seg)

    dictionary={}

    for num in uniques:
        dictionary[num]=0


    time_s=time.time()
    for i in range(0,np.shape(seg)[0]):
        for j in range(0,np.shape(seg)[1]):
            for k in range(0,np.shape(seg)[2]):
                dictionary[seg[i,j,k]]+=1
    time_c=time.time()-time_s
    print("time ",time_c)

    time_s = time.time()
    for i in range(0,np.shape(seg)[0]):
        for j in range(0,np.shape(seg)[1]):
            for k in range(0,np.shape(seg)[2]):
                if(dictionary[seg[i,j,k]]<thresh):
                    seg[i,j,k]=0
    time_c = time.time() - time_s
    print("time ", time_c)

    return new_seg

