import numpy as np


def filter(seg,thresh):

    new_seg=seg

    uniques=np.unique(seg)
    counts=np.zeros(np.shape(uniques)[0])
    n=0
    for num in uniques:
        counts[n]=np.count_nonzero(num==seg)
        n+=1
    kill_list=counts

    for n in range(0,np.shape(counts)[0]):
        if(counts[n]<thresh):
            kill_list[n]=uniques[n]
        else:
            kill_list[n]=0



    kill_list=np.unique(kill_list)
    kill_count=0

    new_seg[(new_seg==kill_list)]=1


    print("Killed %i supervoxels "%np.shape(kill_list)[0])
    return new_seg

