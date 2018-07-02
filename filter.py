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


def top_n(seg,n):

    new_seg = np.zeros(np.shape(seg))

    uniques = np.unique(seg)

    dictionary = {}

    for num in uniques:
        dictionary[num] = 0

    time_s = time.time()
    for i in range(0, np.shape(seg)[0]):
        for j in range(0, np.shape(seg)[1]):
            for k in range(0, np.shape(seg)[2]):
                dictionary[seg[i, j, k]] += 1
    time_c = time.time() - time_s
    print("time ", time_c)


    list=np.zeros(n)
    full_list=np.zeros(np.shape(uniques)[0])
    for i in range(0,np.shape(uniques)[0]):
        full_list[i]=int(dictionary[uniques[i]])

    np.sort(full_list)

    list=full_list[0:n]



    time_s = time.time()
    for i in range(0, np.shape(seg)[0]):
        for j in range(0, np.shape(seg)[1]):
            for k in range(0, np.shape(seg)[2]):
                if (np.equal(dictionary[seg[i, j, k]],list).any()):
                    new_seg[i, j, k] =seg[i,j,k]
    time_c = time.time() - time_s
    print("time ", time_c)


    return new_seg