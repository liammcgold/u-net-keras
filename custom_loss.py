import tensorflow as tf

weight=0


class loss(object):

    def __init__(self):
        self.weights=[1,1,1]

    def set_weight(self,new_weights):
        #takes input as positive # vs negative # for each three affs, ex:
        # weights[0] = [4.77092571, 0.55853532]
        # weights[1] = [25.34738088, 0.51006142]
        # weights[2] = [27.62890733, 0.50921526]

        self.weights[0]=new_weights[0,1]/new_weights[0,0]
        self.weights[1]=new_weights[1,1]/new_weights[1,0]
        self.weights[2]=new_weights[2,1]/new_weights[2,0]

    def print_weight(self):
        print(self.weights)

    def weighted_cross(self,target, logits):
        one_two=tf.add(tf.nn.weighted_cross_entropy_with_logits(targets=target[:,:,:,:,0],logits=logits[:,:,:,:,0],pos_weight=self.weights[0]),
                       tf.nn.weighted_cross_entropy_with_logits(targets=target[:,:,:,:,1],logits=logits[:,:,:,:,1],pos_weight=self.weights[1]))


        one_two_three=tf.add(one_two,tf.nn.weighted_cross_entropy_with_logits(targets=target[:,:,:,:,2],logits=logits[:,:,:,:,2],pos_weight=self.weights[2]))

        return one_two_three