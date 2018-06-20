import numpy as np
import affinities as af
import waterz as wz
import malis as mal






gt=np.load("data/spir_gt.npy")


sample=gt[0:100,0:400,0:400]

nhood=mal.mknhood3d(1)

aff=mal.seg_to_affgraph(sample,nhood)

num_act=np.shape(np.unique(sample))[0]-1

aff=np.asarray(aff,dtype=np.float32)

seg=wz.agglomerate(aff,thresholds=[1])

for segmentation in seg:
    seg=segmentation

num_calc=np.shape(np.unique(seg))[0]-1

print("Calculated: %i"%num_calc)
print("Actual: %i"%num_act)

#print(np.unique(seg))
#print(np.unique(sample))

if(np.equal(seg,sample).all):
    "PASSED"