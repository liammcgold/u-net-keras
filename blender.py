import numpy as np

def blend(arr_1,arr_2,dir,overlap):
    #INPUT MUST BE ZXY

    '''
    DIRECTION INDICATES WHERE arr_2 is to be appended
    EX:
            "x-"

        arr2->arr1

            "x+"

        arr1<-arr2

            "y-"

        arr1
          ^
          |
        arr2

            "y+"

        arr2
          |
          \/
        arr1

    '''

    #write code to work in x+ direction and when x- is activated just switch arrays

    #make sure there are no issues
    arr_1=np.asarray(arr_1.copy())
    arr_2=np.asarray(arr_2.copy())


    #must be same size
    assert (np.shape(arr_1)==np.shape(arr_2)), "MISMATCH OF ARRAY SIZES"



    possible=np.asarray(["z+","z-","x+","x-","y+","y-"])

    assert (dir==possible).any, "INVALID DIRECTION"

    arr_b=np.asarray(9999999999)

    if(dir=="z+"):
        arr_1=np.einsum("czxy->cxzy",arr_1)
        arr_2=np.einsum("czxy->cxzy",arr_2)
        arr_b=xblend(arr_1,arr_2,overlap)
        arr_b=np.einsum("cxzy->czxy",arr_b)
    if(dir=="z-"):
        arr_1 = np.einsum("czxy->cxzy", arr_1)
        arr_2 = np.einsum("czxy->cxzy", arr_2)
        arr_b = xblend(arr_2, arr_1, overlap)
        arr_b = np.einsum("cxzy->czxy", arr_b)
    if(dir=="x+"):
        arr_b=xblend(arr_1,arr_2,overlap)
    if(dir=="x-"):
        arr_b=xblend(arr_2,arr_1,overlap)
    if(dir=="y+"):
        arr_1 = np.einsum("czxy->czyx", arr_1)
        arr_2 = np.einsum("czxy->czyx", arr_2)
        arr_b = xblend(arr_1, arr_2, overlap)
        arr_b = np.einsum("czyx->czxy", arr_b)
    if(dir=="y-"):
        arr_1 = np.einsum("czxy->czyx", arr_1)
        arr_2 = np.einsum("czxy->czyx", arr_2)
        arr_b = xblend(arr_2, arr_1, overlap)
        arr_b = np.einsum("czyx->czxy", arr_b)

    return arr_b


def xblend(arr_1,arr_2,overlap):

    #define
    x_size = (2 - overlap) * np.shape(arr_1)[2]
    overlap_size = int(np.shape(arr_1)[2] * overlap)

    # generate new array
    arr_b = np.zeros((np.shape(arr_1)[0],np.shape(arr_1)[1], int(x_size), np.shape(arr_1)[3]))
    # grab sections that overlap
    ovrlp_1 = arr_1[:,:, -int(overlap_size):, :].copy()
    ovrlp_2 = arr_2[:,:, 0:int(overlap_size), :].copy()

    for x in range(0, int(overlap_size)):
        ovrlp_1[:,:, x, :] = ((overlap_size - x) / overlap_size) * ovrlp_1[:,:, x, :]
        ovrlp_2[:,:, x, :] = ((x) / overlap_size) * ovrlp_2[:,:, x, :]
    ovrlp_comb = np.add(ovrlp_1, ovrlp_2)

    # where does overlap begin index
    point_1 = np.shape(arr_1)[1] - overlap_size - 1
    point_2 = np.shape(arr_1)[1] - 1
    point_3 = overlap_size - 1

    # add front 0->p1
    arr_b[:,:, 0:point_1, :] = arr_1[:,:, 0:point_1, :]
    # add middle p1->p2
    arr_b[:,:, point_1:point_2, :] = ovrlp_comb
    # add end p2->end
    arr_b[:,:, point_2:, :] = arr_2[:,:, point_3:, :]

    return arr_b