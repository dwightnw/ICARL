# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 11:32:59 2022

@author: valentin
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


dataset='mnist'
normalize= True  #to normalize input images
batch_size = 50
one_hot_encode=False


num_classes=10
input_shape = 784
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


#normalize or not normalize the data
if normalize is True:
    x_train=x_train/255
    x_test=x_test/255

# Reserve 10,000 samples for validation.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]


x_train = x_train.astype("float32") 
x_test = x_test.astype("float32") 
x_val = x_val.astype("float32") 

x_train=layers.Flatten()(x_train)
x_val=layers.Flatten()(x_val)
x_test=layers.Flatten()(x_test)

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)


#y_train_sorted_args=np.argsort(-np.array(y_train), kind='mergesort')
#y_train=np.sort(y_train)
#x_train=x_train[y_train_sorted_args,:,:]
y_train_sorted_args_0=np.where(y_train == 0)
y_train_sorted_args_1=np.where(y_train == 1)
y_train_sorted_args_2=np.where(y_train == 2)
y_train_sorted_args_3=np.where(y_train == 3)
y_train_sorted_args_4=np.where(y_train == 4)
y_train_sorted_args_5=np.where(y_train == 5)
y_train_sorted_args_6=np.where(y_train == 6)
y_train_sorted_args_7=np.where(y_train == 7)
y_train_sorted_args_8=np.where(y_train == 8)
y_train_sorted_args_9=np.where(y_train == 9)

x_train_0=tf.squeeze(tf.gather(x_train, y_train_sorted_args_0))
x_train_1=tf.squeeze(tf.gather(x_train, y_train_sorted_args_1))
x_train_2=tf.squeeze(tf.gather(x_train, y_train_sorted_args_2))
x_train_3=tf.squeeze(tf.gather(x_train, y_train_sorted_args_3))
x_train_4=tf.squeeze(tf.gather(x_train, y_train_sorted_args_4))
x_train_5=tf.squeeze(tf.gather(x_train, y_train_sorted_args_5))
x_train_6=tf.squeeze(tf.gather(x_train, y_train_sorted_args_6))
x_train_7=tf.squeeze(tf.gather(x_train, y_train_sorted_args_7))
x_train_8=tf.squeeze(tf.gather(x_train, y_train_sorted_args_8))
x_train_9=tf.squeeze(tf.gather(x_train, y_train_sorted_args_9))

y_train_0=np.full( (x_train_0.shape)[0],0)
y_train_1=np.full( (x_train_1.shape)[0],1)
y_train_2=np.full( (x_train_2.shape)[0],2)
y_train_3=np.full( (x_train_3.shape)[0],3)
y_train_4=np.full( (x_train_4.shape)[0],4)
y_train_5=np.full( (x_train_5.shape)[0],5)
y_train_6=np.full( (x_train_6.shape)[0],6)
y_train_7=np.full( (x_train_7.shape)[0],7)
y_train_8=np.full( (x_train_8.shape)[0],8)
y_train_9=np.full( (x_train_9.shape)[0],9)

y_val_sorted_args_0=np.where(y_val == 0)
y_val_sorted_args_1=np.where(y_val == 1)
y_val_sorted_args_2=np.where(y_val == 2)
y_val_sorted_args_3=np.where(y_val == 3)
y_val_sorted_args_4=np.where(y_val == 4)
y_val_sorted_args_5=np.where(y_val == 5)
y_val_sorted_args_6=np.where(y_val == 6)
y_val_sorted_args_7=np.where(y_val == 7)
y_val_sorted_args_8=np.where(y_val == 8)
y_val_sorted_args_9=np.where(y_val == 9)

x_val_0=tf.squeeze(tf.gather(x_val, y_val_sorted_args_0))
x_val_1=tf.squeeze(tf.gather(x_val, y_val_sorted_args_1))
x_val_2=tf.squeeze(tf.gather(x_val, y_val_sorted_args_2))
x_val_3=tf.squeeze(tf.gather(x_val, y_val_sorted_args_3))
x_val_4=tf.squeeze(tf.gather(x_val, y_val_sorted_args_4))
x_val_5=tf.squeeze(tf.gather(x_val, y_val_sorted_args_5))
x_val_6=tf.squeeze(tf.gather(x_val, y_val_sorted_args_6))
x_val_7=tf.squeeze(tf.gather(x_val, y_val_sorted_args_7))
x_val_8=tf.squeeze(tf.gather(x_val, y_val_sorted_args_8))
x_val_9=tf.squeeze(tf.gather(x_val, y_val_sorted_args_9))

y_val_0=np.full( (x_val_0.shape)[0],0)
y_val_1=np.full( (x_val_1.shape)[0],1)
y_val_2=np.full( (x_val_2.shape)[0],2)
y_val_3=np.full( (x_val_3.shape)[0],3)
y_val_4=np.full( (x_val_4.shape)[0],4)
y_val_5=np.full( (x_val_5.shape)[0],5)
y_val_6=np.full( (x_val_6.shape)[0],6)
y_val_7=np.full( (x_val_7.shape)[0],7)
y_val_8=np.full( (x_val_8.shape)[0],8)
y_val_9=np.full( (x_val_9.shape)[0],9)

y_test_sorted_args_0=np.where(y_test == 0)
y_test_sorted_args_1=np.where( (y_test == 1))
y_test_sorted_args_2=np.where( (y_test == 2)  )
y_test_sorted_args_3=np.where(    (y_test == 3) )
y_test_sorted_args_4=np.where(  (y_test == 4))
y_test_sorted_args_5=np.where( (y_test == 5))
y_test_sorted_args_6=np.where(  (y_test == 6))
y_test_sorted_args_7=np.where(  (y_test == 7))
y_test_sorted_args_8=np.where(  (y_test == 8))
y_test_sorted_args_9=np.where(  (y_test == 9))

x_test_0=tf.squeeze(tf.gather(x_test, y_test_sorted_args_0))
x_test_1=tf.squeeze(tf.gather(x_test, y_test_sorted_args_1))
x_test_2=tf.squeeze(tf.gather(x_test, y_test_sorted_args_2))
x_test_3=tf.squeeze(tf.gather(x_test, y_test_sorted_args_3))
x_test_4=tf.squeeze(tf.gather(x_test, y_test_sorted_args_4))
x_test_5=tf.squeeze(tf.gather(x_test, y_test_sorted_args_5))
x_test_6=tf.squeeze(tf.gather(x_test, y_test_sorted_args_6))
x_test_7=tf.squeeze(tf.gather(x_test, y_test_sorted_args_7))
x_test_8=tf.squeeze(tf.gather(x_test, y_test_sorted_args_8))
x_test_9=tf.squeeze(tf.gather(x_test, y_test_sorted_args_9))

y_test_0=np.full( (x_test_0.shape)[0],0)
y_test_1=np.full( (x_test_1.shape)[0],1)
y_test_2=np.full( (x_test_2.shape)[0],2)
y_test_3=np.full( (x_test_3.shape)[0],3)
y_test_4=np.full( (x_test_4.shape)[0],4)
y_test_5=np.full( (x_test_5.shape)[0],5)
y_test_6=np.full( (x_test_6.shape)[0],6)
y_test_7=np.full( (x_test_7.shape)[0],7)
y_test_8=np.full( (x_test_8.shape)[0],8)
y_test_9=np.full( (x_test_9.shape)[0],9)




# Prepare the training dataset.
train_dataset_0 = tf.data.Dataset.from_tensor_slices((x_train_0, y_train_0))
train_dataset_0 = train_dataset_0.shuffle(buffer_size=1024).batch(batch_size)
train_dataset_1 = tf.data.Dataset.from_tensor_slices((x_train_1, y_train_1))
train_dataset_1 = train_dataset_1.shuffle(buffer_size=1024).batch(batch_size)
train_dataset_2 = tf.data.Dataset.from_tensor_slices((x_train_2, y_train_2))
train_dataset_2 = train_dataset_2.shuffle(buffer_size=1024).batch(batch_size)
train_dataset_3 = tf.data.Dataset.from_tensor_slices((x_train_3, y_train_3))
train_dataset_3 = train_dataset_3.shuffle(buffer_size=1024).batch(batch_size)
train_dataset_4 = tf.data.Dataset.from_tensor_slices((x_train_4, y_train_4))
train_dataset_4 = train_dataset_4.shuffle(buffer_size=1024).batch(batch_size)
train_dataset_5 = tf.data.Dataset.from_tensor_slices((x_train_5, y_train_5))
train_dataset_5 = train_dataset_5.shuffle(buffer_size=1024).batch(batch_size)
train_dataset_6 = tf.data.Dataset.from_tensor_slices((x_train_6, y_train_6))
train_dataset_6 = train_dataset_6.shuffle(buffer_size=1024).batch(batch_size)
train_dataset_7 = tf.data.Dataset.from_tensor_slices((x_train_7, y_train_7))
train_dataset_7 = train_dataset_7.shuffle(buffer_size=1024).batch(batch_size)
train_dataset_8 = tf.data.Dataset.from_tensor_slices((x_train_8, y_train_8))
train_dataset_8 = train_dataset_8.shuffle(buffer_size=1024).batch(batch_size)
train_dataset_9 = tf.data.Dataset.from_tensor_slices((x_train_9, y_train_9))
train_dataset_9 = train_dataset_9.shuffle(buffer_size=1024).batch(batch_size)


# Prepare the validation dataset.
val_dataset_0 = tf.data.Dataset.from_tensor_slices((x_val_0, y_val_0))
val_dataset_0 = val_dataset_0.shuffle(buffer_size=1024).batch(batch_size)
val_dataset_1 = tf.data.Dataset.from_tensor_slices((x_val_1, y_val_1))
val_dataset_1 = val_dataset_1.shuffle(buffer_size=1024).batch(batch_size)
val_dataset_2 = tf.data.Dataset.from_tensor_slices((x_val_2, y_val_2))
val_dataset_2 = val_dataset_2.shuffle(buffer_size=1024).batch(batch_size)
val_dataset_3 = tf.data.Dataset.from_tensor_slices((x_val_3, y_val_3))
val_dataset_3 = val_dataset_3.shuffle(buffer_size=1024).batch(batch_size)
val_dataset_4 = tf.data.Dataset.from_tensor_slices((x_val_4, y_val_4))
val_dataset_4 = val_dataset_4.shuffle(buffer_size=1024).batch(batch_size)
val_dataset_5 = tf.data.Dataset.from_tensor_slices((x_val_5, y_val_5))
val_dataset_5 = val_dataset_5.shuffle(buffer_size=1024).batch(batch_size)
val_dataset_6 = tf.data.Dataset.from_tensor_slices((x_val_6, y_val_6))
val_dataset_6 = val_dataset_6.shuffle(buffer_size=1024).batch(batch_size)
val_dataset_7 = tf.data.Dataset.from_tensor_slices((x_val_7, y_val_7))
val_dataset_7 = val_dataset_7.shuffle(buffer_size=1024).batch(batch_size)
val_dataset_8 = tf.data.Dataset.from_tensor_slices((x_val_8, y_val_8))
val_dataset_8 = val_dataset_8.shuffle(buffer_size=1024).batch(batch_size)
val_dataset_9 = tf.data.Dataset.from_tensor_slices((x_val_9, y_val_9))
val_dataset_9 = val_dataset_9.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the test dataset.
test_dataset_0 = tf.data.Dataset.from_tensor_slices((x_test_0, y_test_0))
test_dataset_0 = test_dataset_0.shuffle(buffer_size=1024).batch(batch_size)
test_dataset_1 = tf.data.Dataset.from_tensor_slices((x_test_1, y_test_1))
test_dataset_1 = test_dataset_1.shuffle(buffer_size=1024).batch(batch_size)
test_dataset_2 = tf.data.Dataset.from_tensor_slices((x_test_2, y_test_2))
test_dataset_2 = test_dataset_2.shuffle(buffer_size=1024).batch(batch_size)
test_dataset_3 = tf.data.Dataset.from_tensor_slices((x_test_3, y_test_3))
test_dataset_3 = test_dataset_3.shuffle(buffer_size=1024).batch(batch_size)
test_dataset_4 = tf.data.Dataset.from_tensor_slices((x_test_4, y_test_4))
test_dataset_4 = test_dataset_4.shuffle(buffer_size=1024).batch(batch_size)
test_dataset_5 = tf.data.Dataset.from_tensor_slices((x_test_5, y_test_5))
test_dataset_5 = test_dataset_5.shuffle(buffer_size=1024).batch(batch_size)
test_dataset_6 = tf.data.Dataset.from_tensor_slices((x_test_6, y_test_6))
test_dataset_6 = test_dataset_6.shuffle(buffer_size=1024).batch(batch_size)
test_dataset_7 = tf.data.Dataset.from_tensor_slices((x_test_7, y_test_7))
test_dataset_7 = test_dataset_7.shuffle(buffer_size=1024).batch(batch_size)
test_dataset_8 = tf.data.Dataset.from_tensor_slices((x_test_8, y_test_8))
test_dataset_8 = test_dataset_8.shuffle(buffer_size=1024).batch(batch_size)
test_dataset_9 = tf.data.Dataset.from_tensor_slices((x_test_9, y_test_9))
test_dataset_9 = test_dataset_9.shuffle(buffer_size=1024).batch(batch_size)

