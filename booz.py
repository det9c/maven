import time
import matplotlib.pyplot as plt
import math
import sys
import scipy
import numpy as np
import random
import os
import pandas as pd
from pandas import DataFrame
from numpy import linalg as la
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, Nadam
from skimage import io
from skimage.transform import resize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import xml.etree.ElementTree as ET
from meter import meter

image_names=[]
with open("aeroplane_train.txt") as file:
    image_names = file.readlines()

nimages=len(image_names)
these_pics=[]
for i in range(0,nimages):
    tmp=[]
    tmp=image_names[i].split()
    if(int(tmp[1])==1):
        these_pics.append(tmp[0])
    
print(len(these_pics))
image_names=[]
with open("aeroplane_val.txt") as file:
    image_names = file.readlines()

nimages=len(image_names)
for i in range(0,nimages):
    tmp=[]
    tmp=image_names[i].split()
    if(int(tmp[1])==1):
        these_pics.append(tmp[0])



nimages=len(these_pics)
inew=400
pixel_array=np.zeros((nimages,inew,inew,3))
objects_per_image=np.zeros((nimages))
object_type=[]
box_truth=[]
box_truth_scaled=[]
original_size=np.zeros((nimages,3))
picture_files=[]


for i in range(0,nimages):
   fname="./JPEGImages/"+str(these_pics[i])+str(".jpg")
   picture_files.append(fname)
   tmp=[]
   pixel_array[i]=resize(io.imread(fname),(inew,inew))
   fname="./Annotations/"+str(these_pics[i])+str(".xml")
   tree=ET.parse(fname)
   root=tree.getroot()
   original_size[i,0]=root.find('size').find('height').text
   original_size[i,1]=root.find('size').find('width').text
   original_size[i,2]=root.find('size').find('depth').text
   objects_in_pic=root.findall('object')
#   objects_per_image[i]=len(objects_in_pic)
   tmp=[]
   tmp2=[]
   icount=0
   for object in objects_in_pic:
#      tmp.append(object.find('name').text)
      thing=object.find('name').text
      if(thing=='aeroplane'):
        icount+=1
        xmax=object.find('bndbox').find('xmax').text
        xmin=object.find('bndbox').find('xmin').text
        ymax=object.find('bndbox').find('ymax').text
        ymin=object.find('bndbox').find('ymin').text
        tmp2.append([ymin,ymax,xmin,xmax])
        tmp.append(thing)
        #scale boxes to new image size
        dxlow=int(inew*(float(xmin)-1)/float(original_size[i,1]))
        dxhigh=int(inew*(float(xmax)-1)/float(original_size[i,1]))
        dylow=int(inew*(float(ymin)-1)/float(original_size[i,0]))
        dyhigh=int(inew*(float(ymax)-1)/float(original_size[i,0]))
        #draw boxes by zero'ing pixels
        pixel_array[i,dylow,dxlow:dxhigh,:]=0
        pixel_array[i,dylow:dyhigh,dxlow,:]=0
        pixel_array[i,dyhigh,dxlow:dxhigh,:]=0
        pixel_array[i,dylow:dyhigh,dxhigh,:]=0
        tmp3=[dylow,dyhigh,dxlow,dxhigh]
   box_truth.append(tmp2)
   box_truth_scaled.append(tmp3)
   object_type.append(tmp)
   objects_per_image[i]=icount
   if(objects_per_image[i]>2):
     plt.show(plt.imshow(pixel_array[i]))



   
'''
#   plt.imshow(data_set[i])
#   plt.show()

tree=ET.parse('./Annotations/2008_000291.xml')
root=tree.getroot()
#for size in root.findall('size'):
depth=size.find('depth').text
height=size.find('height').text
width=size.find('width').text
    
for object in root.findall('object'):
    object_type=object.find('name').text
    xmax=object.find('bndbox').find('xmax').text
    xmin=object.find('bndbox').find('xmin').text
    ymax=object.find('bndbox').find('ymax').text
    ymin=object.find('bndbox').find('ymin').text
    print(xmax)
'''







        




'''
def get_anchor(iside_len,ibottom_len,iguess_up,iguess_across,npixels):
    i=0    
    while i<100000: #infinite loop until stopping condition is satisfied
          iend=iguess_up+iside_len
          if(iend > npixels):
              iguess_up=iguess_up-1
          else:
              break
    i=0
    while i<100000: #infinite loop until stopping condition is satisfied
          iend=iguess_across+ibottom_len
          if(iend > npixels):
              iguess_across=iguess_across-1
          else:
              break

    return iguess_up,iguess_across      
          

def get_anchor_sad(iside_len,ibottom_len,iguess_up,iguess_across,npixels):
    i=0
    while i<100000: #infinite loop until stopping condition is satisfied
          iend=iguess_up-iside_len
          if(iend < 0):
              iguess_up=iguess_up+1
          else:
              break
    i=0
    while i<100000: #infinite loop until stopping condition is satisfied
          iend=iguess_across+ibottom_len
          if(iend > npixels):
              iguess_across=iguess_across-1
          else:
              break

    return iguess_up,iguess_across




np.random.seed(100)
npixels=32
nimages=10000
ibaseline=5
ipic_num=0
picture=np.zeros((npixels,npixels,3))
icount=0
data_set=np.zeros((2*nimages,npixels,npixels,3))
categories=np.zeros((2*nimages))
irow=-1
while ipic_num<nimages:
    iside_len=ibaseline+round(np.random.random()*10)
    ibottom_len=ibaseline+round(np.random.random()*10)
    iguess_up=round(np.random.random()*(npixels-1))
    iguess_across=round(np.random.random()*(npixels-1))
    ianchor_a,ianchor_b=get_anchor(iside_len,ibottom_len,iguess_up,iguess_across,(npixels-1))
#    print(str(iside_len)+" "+str(ibottom_len)+" "+str(ianchor_a)+" "+str(ianchor_b))
    picture[:,:,:]=0.0
    picture[:,:,0:3]=0.0
#    print(str(ianchor_a)+" "+str(iside_len))
    hue=[]
    hue=np.random.random(3)
    picture[ianchor_a:(ianchor_a+iside_len-1),ianchor_b,:]=hue
    picture[ianchor_a:(ianchor_a+iside_len-1),(ianchor_b+ibottom_len-1),:]=hue
    picture[ianchor_a+iside_len-1,ianchor_b:(ianchor_b+ibottom_len),:]=hue
    irow+=1
    data_set[irow]=picture
    categories[irow]=0
    icount+=1
#    plt.subplot(5,8,icount)
#    plt.imshow(picture)
    ipic_num+=1

ipic_num=0
while ipic_num<nimages:
    iside_len=ibaseline+round(np.random.random()*10)
    ibottom_len=ibaseline+round(np.random.random()*10)
    iguess_up=round(np.random.random()*(npixels-1))
    iguess_across=round(np.random.random()*(npixels-1))
    ianchor_a,ianchor_b=get_anchor_sad(iside_len,ibottom_len,iguess_up,iguess_across,(npixels-1))
#    print(str(iside_len)+" "+str(ibottom_len)+" "+str(ianchor_a)+" "+str(ianchor_b))
    picture[:,:,:]=0.0
    picture[:,:,0:3]=0.0
#    print(str(ianchor_a)+" "+str(iside_len))
    hue=[]
    hue=np.random.random(3)
    picture[(ianchor_a-iside_len+1):ianchor_a,ianchor_b,:]=hue
    picture[(ianchor_a-iside_len+1):ianchor_a,(ianchor_b+ibottom_len-1),:]=hue
    picture[ianchor_a-iside_len+1,ianchor_b:(ianchor_b+ibottom_len),:]=hue
    irow+=1
    data_set[irow]=picture
    categories[irow]=1
    icount+=1
#    plt.subplot(5,8,icount)
#    plt.imshow(picture)
    ipic_num+=1

plt.show()

icount=0
i=0
while i<40:
  icount+=1
  plt.subplot(5,8,icount)
  plt.imshow(data_set[i])
  i+=1
    
#plt.show()


np.random.seed(100)
train_images, test_images, train_labels, test_labels =train_test_split(data_set,categories,test_size=.1)

ntrain=len(train_images)
scale_values=np.zeros((npixels,npixels,3,2))
i=0
while i<npixels:
    j=0
    while j<npixels:
        k=0
        while k<3:
            scale_values[i,j,k,0]=np.mean(train_images[:,i,j,k])
            scale_values[i,j,k,1]=np.std(train_images[:,i,j,k]  )                   
            train_images[:,i,j,k]=train_images[:,i,j,k]-scale_values[i,j,k,0]
            test_images[:,i,j,k]=test_images[:,i,j,k]-scale_values[i,j,k,0]
            if(scale_values[i,j,k,1] >0.0):
               train_images[:,i,j,k]=train_images[:,i,j,k]/scale_values[i,j,k,1]
               test_images[:,i,j,k]=test_images[:,i,j,k]/scale_values[i,j,k,1]
            k+=1
        j+=1
    i+=1


#MLP
model = keras.Sequential([keras.layers.Flatten(input_shape=(npixels,npixels,3)),
        keras.layers.Dense(npixels*npixels, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


np.random.seed(100)
#CNN
model = keras.Sequential([keras.layers.Conv2D(input_shape=(npixels,npixels,3),filters=8,kernel_size=2,strides=(1,1),padding='valid',data_format="channels_last",use_bias=True,activation=tf.nn.relu),keras.layers.Flatten(),keras.layers.Dense(100, activation=tf.nn.relu),keras.layers.Dense(2, activation=tf.nn.softmax)])



model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


print(model.summary())

history=model.fit(train_images, train_labels, epochs=5)


test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

print(model.metrics_names)
pred=model.predict(test_images)
imax=len(test_images)
i=0
predictions=[]
while i<imax:
    predictions.append(np.argmax(pred[i]))
    i+=1

print(confusion_matrix(predictions,test_labels))


direction=['up','down']
i=0
while i<imax:
  if(predictions[i] != test_labels[i]):
    plt.imshow(test_images[i])
    word=str(direction[predictions[i]])+" "+str(pred[i])
    plt.xlabel(word,fontsize=12,fontweight='bold')
    plt.show()
  i+=1

predictions = model.predict(test_images)
# Plot the first 25 test images, their predicted label, and the true label
# Color correct predictions in green, incorrect predictions in red
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
      color = 'green'
    else:
      color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label], 
                                  class_names[true_label]),
                                  color=color)
plt.show()
    



#dimension math
#
# so the filters always go through the entire activation volume.  not one feature map at a time.
# maxpooling is done by layer though. so if have 4 layers, take max in each of the 4 layers and output 
# depth is still four

npixels=32
model = keras.Sequential([keras.layers.Conv2D(input_shape=(npixels,npixels,3),filters=4,kernel_size=4,strides=(1,1)),keras.layers.Conv2D(filters=3,kernel_size=2,strides=(1,1),use_bias=True),keras.layers.Flatten(),keras.layers.Dense(10)])
'''

