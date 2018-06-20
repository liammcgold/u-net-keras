import numpy as np
import blender as blend
import math


class assembler(object):

    def __init__(self,raw,overlap,shape):
        self.raw=raw
        self.num_blocks,self.padding=calc_num_blocks(np.shape(raw),overlap,shape)




def calc_num_blocks(raw_shape,overlap,shape):

    '''
    calculate raw dim  size using formula:

    size  =  (raw_shape)/(block_size)  *  (1/overlap)  -  1

    round up to get actual number

    calc padding with formula:

    padding  =  block_size/remainder


    padding is rounded up

    '''

    num_blocks_raw=np.zeros(3)
    num_blocks=np.zeros(3)
    padding=np.zeros(3)
    for x in range(0,3):
        num_blocks_raw[x]=(raw_shape[x]/shape[x])*(1/overlap)-1
        num_blocks=math.ceil(num_blocks_raw[x])
        padding=math.ceil()

    #round up to find how many blocks are needed

    num_blocks=math.ceil(num_blocks_dim_1_raw)

