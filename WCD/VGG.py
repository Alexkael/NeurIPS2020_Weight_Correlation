import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import regularizers
import keras.backend as K

import os
from keras.optimizers import SGD

import utils

import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="3" 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
session = tf.Session(config=config)


cfg = {}
cfg['SGD_BATCHSIZE']    = 32
cfg['SGD_LEARNINGRATE'] = 0.01
cfg['NUM_EPOCHS']       = 500

cfg['ACTIVATION'] = 'relu'

trn, tst = utils.get_cifar10('CNN')

model = Sequential()
model.add(Conv2D(4, (3, 3), padding='same',
input_shape=(32,32,3)))
model.add(Activation(cfg['ACTIVATION']))
model.add(BatchNormalization())


model.add(Conv2D(4, (3, 3), padding='same'))
model.add(Activation(cfg['ACTIVATION']))
model.add(BatchNormalization())


model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(4, (3, 3), padding='same'))
model.add(Activation(cfg['ACTIVATION']))
model.add(BatchNormalization())


model.add(Conv2D(4, (3, 3), padding='same'))
model.add(Activation(cfg['ACTIVATION']))
model.add(BatchNormalization())


model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(4, (3, 3), padding='same'))
model.add(Activation(cfg['ACTIVATION']))
model.add(BatchNormalization())

model.add(Conv2D(4, (3, 3), padding='same'))
model.add(Activation(cfg['ACTIVATION']))
model.add(BatchNormalization())


model.add(Conv2D(4, (3, 3), padding='same'))
model.add(Activation(cfg['ACTIVATION']))
model.add(BatchNormalization())

model.add(Conv2D(4, (3, 3), padding='same'))
model.add(Activation(cfg['ACTIVATION']))
model.add(BatchNormalization())


model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, (3, 3), padding='same'))
model.add(Activation(cfg['ACTIVATION']))
model.add(BatchNormalization())

model.add(Conv2D(8, (3, 3), padding='same'))
model.add(Activation(cfg['ACTIVATION']))
model.add(BatchNormalization())


model.add(Conv2D(8, (3, 3), padding='same'))
model.add(Activation(cfg['ACTIVATION']))
model.add(BatchNormalization())

model.add(Conv2D(8, (3, 3), padding='same'))
model.add(Activation(cfg['ACTIVATION']))
model.add(BatchNormalization())


model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, (3, 3), padding='same'))
model.add(Activation(cfg['ACTIVATION']))
model.add(BatchNormalization())

model.add(Conv2D(8, (3, 3), padding='same'))
model.add(Activation(cfg['ACTIVATION']))
model.add(BatchNormalization())


model.add(Conv2D(8, (3, 3), padding='same'))
model.add(Activation(cfg['ACTIVATION']))
model.add(BatchNormalization())

model.add(Conv2D(8, (3, 3), padding='same'))
model.add(Activation(cfg['ACTIVATION']))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(12))
model.add(Activation(cfg['ACTIVATION']))
model.add(BatchNormalization())

model.add(Dense(12))
model.add(Activation(cfg['ACTIVATION']))
model.add(BatchNormalization())

model.add(Dense(10))
model.add(Activation('softmax'))


optimizer = SGD(lr=cfg['SGD_LEARNINGRATE'])


model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy','top_k_categorical_accuracy'])


r = model.fit(x=trn.X, y=trn.Y, 
              verbose    = 2, 
              batch_size = cfg['SGD_BATCHSIZE'],
              epochs     = cfg['NUM_EPOCHS'],
              validation_data=(tst.X, tst.Y),
              callbacks  = [])



