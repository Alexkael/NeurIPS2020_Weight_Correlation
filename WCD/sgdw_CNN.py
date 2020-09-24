from keras.legacy import interfaces
from keras.optimizers import Optimizer
from keras import backend as K

import tensorflow as tf
import numpy as np
import sys

def concat(x,i):
    y = [x]
    for ii in range(i-1):
        y = tf.concat([y, [x]], 0)
    return y

def correlation_gradient(x,nl1,nl2):
    x = tf.transpose(x)
    num = x.get_shape().as_list()

    d_inner = tf.matmul(tf.transpose(x),x) #inner product
    d_inner = tf.reshape(tf.boolean_mask(d_inner,~tf.eye(tf.size(d_inner[0]),dtype=bool)),([tf.size(d_inner[0]),-1]))

    l2 = tf.norm (x,axis=0)
    d_norm = tf.tensordot(l2,l2,axes=0) #l2 norm
    d_norm = tf.reshape(tf.boolean_mask(d_norm,~tf.eye(tf.size(d_norm[0]),dtype=bool)),([tf.size(d_norm[0]),-1]))
    '''
    cc = tf.reduce_mean(d_inner/d_norm)
    co = nl1*(nl2-1)/(1-cc) + nl1*(nl2-1)/(1+(nl2-1)*cc)
    
    def cond1(t1):
        return tf.less(t1, 0.5)

    def body1(t1):
        return t1*10
    
    co = tf.while_loop(cond1, body1, [co])
    
    def cond2(t1):
        return tf.less(5.0, t1)

    def body2(t1):
        return t1/10
    
    co = tf.while_loop(cond2, body2, [co])
    '''
    d_norm2 = tf.tensordot(l2*tf.square(l2),l2,axes=0)
    d_norm2 = tf.reshape(tf.boolean_mask(d_norm2,~tf.eye(tf.size(d_norm2[0]),dtype=bool)),([tf.size(d_norm2[0]),-1]))
    d_norm2 = d_inner/d_norm2

    s1 = [tf.concat([x[:,:0], x[:,1:]], 1)/concat(d_norm[0],num[0])]
    s2 = [tf.tensordot(x[:,0], d_norm2[0], axes=0)]
    s3 = [concat(tf.sign(d_inner[0]),num[0])]

    for i in range(1,num[1]):
        s1 = tf.concat([s1, [tf.concat([x[:,:i], x[:,(i+1):]], 1)/concat(d_norm[i],num[0])]], 0)
        s2 = tf.concat([s2, [tf.tensordot(x[:,i], d_norm2[i], axes=0)]], 0)
        s3 = tf.concat([s3, [concat(tf.sign(d_inner[i]),num[0])]], 0)

    s = s1 - s2
    s = tf.reduce_sum(s*s3,axis=2)/(num[1]-1)
    #return s*co
    return s

class SGD(Optimizer):

    def __init__(self, lr=0.01,**kwargs):
        super(SGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(lr, name='lr')

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params) 
        '''
        iii = 0
        for p, g in zip(params, grads):
            print(iii)
            print(p)
            iii += 1
        sys.exit(0)
        '''
        self.updates = [K.update_add(self.iterations, 1)] 
        self.weights = [self.iterations] 
        iter = 0
        for p, g in zip(params, grads):
            #vgg11
            #if iter%4==0 and iter<19:
            #vgg16
            #if iter%4==0 and iter<39:
            #vgg19
            if iter%4==0 and iter<47:
                x = tf.transpose(p, perm=[2,3,0,1])
                num_ = x.get_shape().as_list()
                x = tf.reshape(x,[num_[0],num_[1],-1])
                c_g = [correlation_gradient(x[0],num_[0],num_[1])]
                for ii in range(1,num_[0]):
                    c_g = tf.concat([c_g, [correlation_gradient(x[ii],num_[0],num_[1])]], 0)
                c_g = tf.reshape(c_g,[num_[0],num_[1],num_[2],num_[3]])
                c_g = tf.transpose(c_g, perm=[2,3,0,1])

                new_p = p - self.learning_rate * g - 0.1 * self.learning_rate * c_g
                #0.4 * (tf.reduce_mean(g)/tf.reduce_mean(c_g)) * self.lr * c_g
            else:
                new_p = p - self.learning_rate * g
            
            #new_p = p - self.learning_rate * g
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            
            self.updates.append(K.update(p, new_p))
            iter += 1
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr))}
        base_config = super(SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
