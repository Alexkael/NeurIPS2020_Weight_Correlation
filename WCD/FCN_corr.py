import keras
import keras.backend as K
import numpy as np

import os
import utils
import loggingreporter 
from sgdw import SGD

import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="3" 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
session = tf.Session(config=config)
    
network_1 = [52,48,44,40,36,32,28,24,20,16]
#network_2 = [10000,1000]

cfg = {}
cfg['SGD_BATCHSIZE']    = 64
cfg['SGD_LEARNINGRATE'] = 0.01
cfg['NUM_EPOCHS']       = 50

cfg['ACTIVATION'] = 'relu'
cfg['LAYER_DIMS'] = network_1

trn, tst = utils.get_mnist('FCN')


input_layer  = keras.layers.Input((trn.X.shape[1],))
clayer = input_layer
for n in cfg['LAYER_DIMS']:
    clayer = keras.layers.Dense(n, activation=cfg['ACTIVATION'])(clayer)
output_layer = keras.layers.Dense(trn.nb_classes, activation='softmax')(clayer)
model = keras.models.Model(inputs=input_layer, outputs=output_layer)

optimizer = SGD(lr=cfg['SGD_LEARNINGRATE'])

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy','top_k_categorical_accuracy'])

r = model.fit(x=trn.X, y=trn.Y, 
              verbose    = 2, 
              batch_size = cfg['SGD_BATCHSIZE'],
              epochs     = cfg['NUM_EPOCHS'],
              validation_data=(tst.X, tst.Y),
              callbacks  = []
              )


