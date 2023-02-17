#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 14:04:34 2022

@author: dnwaigwe
"""


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from load_process_mnist_data_flattened import *
import math
import time
import functools
from collections import Counter



class icarl:
    def __init__(self, K,D, nb_classes):
        self.x_data=None
        self.y_data=None
        self.x_data_val=None
        self.x_data_val=None
        self.P=[] #exemplar sets
        self.K=K
        self.D=D
        self.m=int(K)
        self.s=None
        self.t=None
        self.loss=None
        self.first_class=True
        
        self.g = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Flatten(),
                layers.Dense(300,activation="relu"),
                layers.Dense(D, activation="relu"),
                layers.Dense(nb_classes, activation="sigmoid"),
            ]
        )
        self.g.layers[0]._name='flatten'
        self.g.layers[1]._name='fc1'
        self.g.layers[2]._name='fc2'
        self.g.layers[3]._name='fc3'
        
        self.phi=keras.Model(inputs=self.g.input, outputs=self.g.layers[-2].output)
        

    def classify(self, x):
        mu=[]
        for el in self.P:
            g=functools.partial(np.expand_dims, axis=0)
            avg_feature= np.squeeze(np.array(list(map( self.phi, np.array( list( map(g, el )))  ))))
            avg_feature=np.mean(avg_feature,0)
            mu.append(avg_feature)
        dist=[]
        for i,el in enumerate(mu):
            dist.append(np.linalg.norm( mu[i]- self.phi(np.expand_dims(x,0) )))
        min_dist=min(dist)
        y_star=dist.index(min_dist)
        return y_star

    def compute_loss_initial(self, y,q_y):
        labels=  [ tf.cast(y[j]==i, tf.float32)  for i in range(0,self.s) for j in range(0,y.shape[0])]
        labels=tf.convert_to_tensor(labels)
        logits=functools.reduce(lambda a, b: tf.concat([a,b],0), [ q_y[:,i]  for i in range(0,self.s)  ])
        print(logits)
        self.loss=tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        
        
    def compute_loss(self, y, q_y):
        distillation_loss=0
        classification_loss=0

        labels=  [ tf.cast(y[j]==i, tf.float32)   for i in range(self.s,self.t) for j in range(0,y.shape[0])]
        labels=tf.convert_to_tensor(labels)
        logits=functools.reduce(lambda a, b: tf.concat([a,b],0), [ q_y[:,i]  for i in  range(self.s,self.t) ])
        classification_loss=tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        
        labels=  [q_y[j,i]   for i in range(0,self.s) for j in range(0,y.shape[0])]
        labels=tf.convert_to_tensor(labels)
        logits=functools.reduce(lambda a, b: tf.concat([a,b],0), [ q_y[:,i]  for i in range(0,self.s)  ])
        distillation_loss=tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        print(logits)
        self.loss=(distillation_loss+classification_loss)
        

    def incremental_train_remainder(self, new_samples):
        new_samples=[el[0] for el in new_samples]
        if icarl_object.P !=[]:
            for i in range(0, self.s):
                self.reduce_exemplar_set(i)
            for i in range(0,  len(new_samples)):
                print('Constructing exemplar set: set: %1d  out of %1d' % (i, self.t))
                self.construct_exemplar_set(new_samples[i])
            self.s=self.t
            
        else:
            for i in range(0,  len(new_samples)):
                print('Constructing exemplar set: set: %1d  out of %1d' % (i, self.s))
                self.construct_exemplar_set(new_samples[i])
            #self.P.append(0) 

    def reduce_exemplar_set(self, i):
        self.P[i]= self.P[i][0:self.m]
        
    def construct_exemplar_set(self, X):
        p=[]
        class_mean=np.mean(self.phi(X),0)
        for k in range(0,self.m):
            dist=[]
            sum_phi=0
            for i in range(0,k):
                sum_phi=sum_phi+ self.phi(np.expand_dims(p[i],0))
            for j in range(0,len(X)):
                dist.append(np.linalg.norm( class_mean- ( self.phi(np.expand_dims(X[j],0))+ sum_phi)/(k+1) ))
            min_dist=min(dist)
            p_k=dist.index(min_dist)
            p.append(X[p_k])
        self.P.append(p)
                
                

optimizer = tf.keras.optimizers.Adam(learning_rate=1*1e-2)
acc_metric=tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric=tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        q_y = model(x, training=True)
        if icarl_object.first_class==True:
            icarl_object.compute_loss_initial(y,q_y)
        else:
            icarl_object.compute_loss(y,q_y)
    grads = tape.gradient(icarl_object.loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    if q_y.shape[1] ==1:
        acc_metric.update_state(tf.expand_dims(y,1), q_y)
    else:
        acc_metric.update_state(y, q_y)
    return icarl_object.loss

@tf.function
def test_step(model, x, y):
    val = model(x, training=False)
    if val.shape[1] ==1:
        val_acc_metric.update_state(tf.expand_dims(y,1), val)
    else:
        val_acc_metric.update_state(y, val)

def train(model,epochs,dataset_train, dataset_val):
    if icarl_object.s==None:
        icarl_object.s=len(dataset_train)
        icarl_object.first_class=True
    else:
        s_prime= len(dataset_train)
        icarl_object.t=icarl_object.s+s_prime
        icarl_object.m=int(icarl_object.K/icarl_object.t)
        icarl_object.first_class=False
        
        oldweight=icarl_object.g.layers[-1].get_weights()
        bigger_layer=layers.Dense(icarl_object.t, activation="sigmoid")
        tmp=tf.convert_to_tensor(np.ones((1,icarl_object.D)))
        bigger_out=bigger_layer(tmp)
        bigger_weights=bigger_layer.get_weights()
        bigger_weights[0][:,0:icarl_object.s]=oldweight[0]
        bigger_layer.set_weights(bigger_weights)
        
        icarl_object.g.pop()
        icarl_object.g.add(bigger_layer)
        icarl_object.g.layers[-1].set_weights(bigger_weights)
        icarl_object.phi=keras.Model(inputs=icarl_object.g.input, outputs=icarl_object.g.layers[-2].output)
        icarl_object.g.layers[3]._name='fc3'
        

    #extract data from lists
    xlist=[el[0] for el in dataset_train]
    x_data=functools.reduce(lambda a, b: tf.concat([a,b],0), xlist)
    ylist=[el[1] for el in dataset_train]
    y_data=functools.reduce(lambda a, b: tf.concat([a,b],0), ylist)
    
    #combine new data with old data
    if icarl_object.x_data !=None:
        icarl_object.x_data=tf.concat([icarl_object.x_data, x_data],0)
        icarl_object.y_data=tf.concat([icarl_object.y_data, y_data],0)
    else:
        icarl_object.x_data=x_data
        icarl_object.y_data=y_data

    current_train_dataset = tf.data.Dataset.from_tensor_slices((icarl_object.x_data, icarl_object.y_data))
    current_train_dataset = current_train_dataset.shuffle(buffer_size=10000).batch(batch_size)
    
    xlist=[el[0] for el in dataset_val]
    x_data_val=functools.reduce(lambda a, b: tf.concat([a,b],0), xlist)
    ylist=[el[1] for el in dataset_val]
    y_data_val=functools.reduce(lambda a, b: tf.concat([a,b],0), ylist)
    
    if icarl_object.x_data_val !=None:
        icarl_object.x_data_val=tf.concat([icarl_object.x_data_val, x_data_val],0)
        icarl_object.y_data_val=tf.concat([icarl_object.y_data_val, y_data_val],0)
    else:
        icarl_object.x_data_val=x_data_val
        icarl_object.y_data_val=y_data_val
    
    current_val_dataset = tf.data.Dataset.from_tensor_slices((icarl_object.x_data_val, icarl_object.y_data_val))
    current_val_dataset = current_val_dataset.shuffle(buffer_size=10000).batch(batch_size)
    
    
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
         
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(current_train_dataset):
            loss_value = train_step(model,x_batch_train, y_batch_train)
    
            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * batch_size))
    
        # Display metrics at the end of each epoch.
        train_acc = acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
    
        # Reset training metrics at the end of each epoch
        acc_metric.reset_states()
    
        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in current_val_dataset:
            test_step(model,x_batch_val, y_batch_val)
    
        val_acc = acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))
   
K=50
D=100
num_epoch=10

datalist_train=[ (x_train_0,y_train_0)       ]
datalist_val=[ (x_val_0,y_val_0)      ]
icarl_object=icarl(K,D,len(datalist_train))
train(icarl_object.g, num_epoch, datalist_train,datalist_val)
icarl_object.incremental_train_remainder(datalist_train)

datalist_train=[ (x_train_1,y_train_1)         ]
datalist_val=[ (x_val_1,y_val_1)       ]
train(icarl_object.g, num_epoch, datalist_train,datalist_val)
icarl_object.incremental_train_remainder(datalist_train)
icarl_object.classify(x_train_1[0])

datalist_train=[(x_train_2,y_train_2)        ]
datalist_val=[ (x_val_2,y_val_2)       ]
train(icarl_object.g,num_epoch, datalist_train,datalist_val)
icarl_object.incremental_train_remainder(datalist_train)
icarl_object.classify(x_train_2[0])

datalist_train=[(x_train_3,y_train_3)          ]
datalist_val=[ (x_val_3,y_val_3)         ]
train(icarl_object.g, num_epoch, datalist_train,datalist_val)
icarl_object.incremental_train_remainder(datalist_train)
icarl_object.classify(x_train_3[0])

datalist_train=[(x_train_3,y_train_3)          ]
datalist_val=[ (x_val_3,y_val_3)         ]
train(icarl_object.g, num_epoch, datalist_train,datalist_val)
icarl_object.incremental_train_remainder(datalist_train)
icarl_object.classify(x_train_3[0])

datalist_train=[(x_train_4,y_train_4)          ]
datalist_val=[ (x_val_4,y_val_4)         ]
train(icarl_object.g, num_epoch, datalist_train,datalist_val)
icarl_object.incremental_train_remainder(datalist_train)
icarl_object.classify(x_train_4[0])

datalist_train=[(x_train_5,y_train_5)          ]
datalist_val=[ (x_val_5,y_val_5)         ]
train(icarl_object.g, num_epoch, datalist_train,datalist_val)
icarl_object.incremental_train_remainder(datalist_train)
icarl_object.classify(x_train_5[0])

datalist_train=[(x_train_6,y_train_6)          ]
datalist_val=[ (x_val_6,y_val_6)         ]
train(icarl_object.g, num_epoch, datalist_train,datalist_val)
icarl_object.incremental_train_remainder(datalist_train)
icarl_object.classify(x_train_6[0])

datalist_train=[(x_train_7,y_train_7)          ]
datalist_val=[ (x_val_7,y_val_7)         ]
train(icarl_object.g, num_epoch, datalist_train,datalist_val)
icarl_object.incremental_train_remainder(datalist_train)
icarl_object.classify(x_train_7[0])

datalist_train=[(x_train_8,y_train_8)          ]
datalist_val=[ (x_val_8,y_val_8)         ]
train(icarl_object.g, num_epoch, datalist_train,datalist_val)
icarl_object.incremental_train_remainder(datalist_train)
icarl_object.classify(x_train_8[0])

datalist_train=[(x_train_9,y_train_9)          ]
datalist_val=[ (x_val_9,y_val_9)         ]
train(icarl_object.g, num_epoch, datalist_train,datalist_val)
icarl_object.incremental_train_remainder(datalist_train)
icarl_object.classify(x_train_9[0])
