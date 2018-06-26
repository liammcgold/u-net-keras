import assembler
import numpy as np
import tifffile as tif
import code


a=np.reshape(np.zeros((5,4,6)),(5,4,6))

b=np.ones((2,2,2))

b[0,1,1]=3

a[1:3,0:2,0:2]=b





#num_blocks, padding=assembler.calc_num_blocks(np.shape(a),.5,(2,2,2))



#print(b)

#print(assembler.get_block(,num_blocks,a,(2,2,2),.5))



#Get block looks good and so does the edge detector




raw=np.load("data/spir_raw.npy")

raw=raw[0:25,0:250,0:250]



raw=np.einsum("zxy->xyz",raw)

#blending on z axis is bad (2 axis)
#y axis is 2 axis is verticle on images
#x axis is 0 axis and horizontal on image



def function(input):
    return np.asarray([input,input,input])

builder=assembler.assembler(raw,.5,(128,128,16),function)



aff=builder.process()

#code.interact(local=locals())

raw=np.einsum("xyz->zxy",raw)
aff=np.einsum("cxyz->czxy",aff)


assert np.equal(raw,aff[0]).all()


for i in range(0,np.shape(aff)[1]):
    tif.imsave("assembler_testing/processed/proc%i"%i,np.asarray(aff[0,i,:,:],dtype=np.float32))
    tif.imsave("assembler_testing/act/act%i"%i,np.asarray(raw[i,:,:],dtype=np.float32))
