import numpy as np
import waterz as w
import tifffile as tif


import filter as fil


thresh=0.1
thresh_l=0.0000001
thresh_h=0.99999999999

filter_thresh=200


pred=np.load("data_out/prediction.npy")

aff=np.load("/home/user1/code/Cluster/u-net-keras/data/spir_aff.npy")

gt=np.load("/home/user1/code/Cluster/u-net-keras/data/spir_gt.npy")


pred=pred[0]

pred=np.einsum("zxya->azxy",pred)


temp0=pred[0]
temp1=pred[1]
temp2=pred[2]

pred[0]=temp2
pred[1]=temp1
pred[2]=temp0

aff=np.asarray(aff,dtype=np.float32)
pred=np.asarray(pred,dtype=np.float32)
gt=np.asarray(gt,dtype=np.uint32)

seg=w.agglomerate(pred,thresholds=[thresh],gt=gt, aff_threshold_low=thresh_l,aff_threshold_high=thresh_h)


print("predicted averages: ",np.average(pred[0]),np.average(pred[1]),np.average(pred[2]))
print("actual averages: ",np.average(aff[0]),np.average(aff[1]),np.average(aff[2]))

for segmentation in seg:
    seg=segmentation

seg=seg[0]


print("FOUND %i UNIQUE NEURONS"%(np.shape(np.unique(seg))[0]-2))

truth_table=0

n=0
for i in range(0,np.shape(seg)[0]):
    if np.shape(np.unique(seg[i]))[0]!=1:
        truth_table=1

    n+=1



if truth_table==0:
    print("BAD SEGMENTATION")

#seg=fil.filter(seg,filter_thresh)




for i in range(0,np.shape(seg)[0]):
    tif.imsave("watershed_tiffs/pred0%i.tiff"%i,np.asarray(seg[i],dtype=np.float32))