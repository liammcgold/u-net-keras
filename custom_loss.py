import tensorflow as tf


weight=0


class loss(object):

    def __init__(self):
        self.weight=1

    def set_weight(self,new_weight):
        self.weight=new_weight

    def print_weight(self):
        print(self.weight)

    def weighted_cross(self,target, logits):
        return tf.nn.weighted_cross_entropy_with_logits(targets=target,logits=logits,pos_weight=self.weight)