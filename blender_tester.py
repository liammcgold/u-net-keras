import blender
import numpy as np

all_ones=np.ones((3,20,20,20))

all_zeros=np.zeros((3,20,20,20))

directions=["z+","z-","x+","x-","y+","y-"]

for dir in directions:

    print("testing "+dir+" direction...")

    blended_ones=blender.blend(all_ones,all_ones,dir,.5)

    assert (blended_ones==1).all(), "BLENDING OF TWO ALL 1 ARRAYS CONTAINS NON 1 VALUE"
    print("     ALL ONES PASSED")

    blended_zeros=blender.blend(all_zeros,all_zeros,dir,.1)

    assert (blended_zeros==0).all(), "BLENDING OF TWO ALL 0 ARRAYS CONTAINS NON 0 VALUE"
    print("     ALL ZEROS PASSED")


    if(dir=="z+" or dir=="z-"):

        assert (np.shape(blended_ones)==(3,30,20,20)), "BLENDING SIZE IS NOT ACCURATE IN Z, IS %s"%(np.shape(blended_ones),)+"INSTEAD OF %S"%((3,30,20,20),)
        assert (np.shape(blended_zeros)==(3,38,20,20)), "BLENDING SIZE IS NOT ACCURATE IN Z, IS %s"%(np.shape(blended_zeros),)+"INSTEAD OF %S"%((3,38,20,20),)

    if(dir=="x+" or dir=="x-"):
        assert (np.shape(blended_ones) == (3, 20, 30, 20)), "BLENDING SIZE IS NOT ACCURATE IN X, IS %s" % (np.shape(blended_ones),) + "INSTEAD OF %S" % ((3, 30, 20, 20),)
        assert (np.shape(blended_zeros) == (3, 20, 38, 20)), "BLENDING SIZE IS NOT ACCURATE IN X, IS %s" % (np.shape(blended_zeros),) + "INSTEAD OF %S" % ((3, 38, 20, 20),)

    if(dir=="y+" or dir=="=y-"):
        assert (np.shape(blended_ones) == (3, 20, 20, 30)), "BLENDING SIZE IS NOT ACCURATE IN Y, IS %s" % (np.shape(blended_ones),) + "INSTEAD OF %S" % ((3, 20, 20, 30),)
        assert (np.shape(blended_zeros) == (3, 20, 20, 38)), "BLENDING SIZE IS NOT ACCURATE IN Y, IS %s" % (np.shape(blended_zeros),) + "INSTEAD OF %S" % ((3, 20, 20, 38),)
    print("     SHAPES PASSED")
