import assembler
import numpy as np
import tifffile as tif


a=np.reshape(np.zeros((5,4,6)),(5,4,6))

b=np.ones((2,2,2))

b[0,1,1]=3

a[1:3,0:2,0:2]=b





#num_blocks, padding=assembler.calc_num_blocks(np.shape(a),.5,(2,2,2))



#print(b)

#print(assembler.get_block(,num_blocks,a,(2,2,2),.5))



#Get block looks good and so does the edge detector



raw=np.load("data/spir_raw.npy")


raw=np.einsum("zxy->xyz",raw)


def function(input):
    return input

assembler=assembler.assembler(raw,.5,(128,128,16),function)


for i in range(0,400):
    raw_slice=np.asarray(assembler.feed(),dtype=np.float32)
    tif.imsave("assembler_testing/slice%i"%i,raw_slice[:,:,0])