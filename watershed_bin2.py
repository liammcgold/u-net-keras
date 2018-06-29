import numpy as np
import waterz as w
import tifffile as tif
import add_colors
import filter


thresh=0.3
thresh_l=0.000001
thresh_h=0.9999999

filter_thresh=500

itteration=900000

aff=np.load("/home/user1/code/Cluster/u-net-keras/data/spir_bin2_aff%i.npy"%itteration)

gt=np.load("/home/user1/code/Cluster/u-net-keras/data/spir_gt.npy")

#aff=np.einsum("azxy->azxy",aff)

aff=np.asarray(aff,dtype=np.float32)
gt=np.asarray(gt,dtype=np.uint32)


aff=aff[:,0:32,:,:-50]

#aff=aff[:,0:10,:,:]
gt=gt[0:32,:,:-50]


raw=np.load("/home/user1/code/Cluster/u-net-keras/data/spir_raw.npy")


raw=raw[:,:,:-50]
#
# ######################
# model=u_net_MG.make()
#
# itteration=800000
#
# model.load_weights("saved_models_MG/model%i"%itteration)
#
# raw_slice=random_provider.random_provider_raw((16,128,128),raw)
#
# raw_slice=np.reshape(raw_slice,(1,16,128,128,1))
#
# aff=model.predict(raw_slice)
#
# aff=np.einsum("bzxyc->bczxy",aff)
#
# aff=aff[0]
#
# print(np.shape(aff))
#
# ##########################



seg=w.agglomerate(aff,thresholds=[thresh],gt=gt)#, aff_threshold_low=thresh_l,aff_threshold_high=thresh_h,scoring_function="OneMinus<MinAffinity<RegionGraphType, ScoreValue>>")



for segmentation in seg:
    seg=segmentation


seg=seg[0]


print("FOUND %i UNIQUE NEURONS"%(np.shape(np.unique(seg))[0]-2))

seg=filter.filter(seg,400)

seg=add_colors.add_colors(seg)

print(np.shape(seg))

for i in range(0,np.shape(seg)[0]):
    tif.imsave("bin2 _prediction/pred0/pred0%i.tiff"%i,np.asarray(seg[i],dtype=np.float16),photometric='rgb')
    tif.imsave("bin2 _prediction/raw/raw%i.tiff"%i,np.asarray(raw[i,:,:],dtype=np.float32))
    tif.imsave("bin2 _prediction/gt/gt%i.tiff"%i,np.asarray(gt[i],dtype=np.float32))
    tif.imsave("bin2 _prediction/aff_graph0/0aff%i.tiff"%i,np.asarray(aff[0,i],dtype=np.float32))
    tif.imsave("bin2 _prediction/aff_graph1/1aff%i.tiff"%i,np.asarray(aff[1,i],dtype=np.float32))
    tif.imsave("bin2 _prediction/aff_graph2/2aff%i.tiff"%i,np.asarray(aff[2,i],dtype=np.float32))

#'''