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
import progressbar


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
nimages=10
inew=400
pixel_array=np.zeros((nimages,inew,inew,3))
objects_per_image=np.zeros((nimages))
object_type=[]
box_truth=[]
box_truth_scaled=[]
original_size=np.zeros((nimages,3))
picture_files=[]

#nimages=10
for i in progressbar.progressbar(range(0,nimages)):
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
   tmp3=[]
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
        tmp3.append([dylow,dyhigh,dxlow,dxhigh])
   box_truth.append(tmp2)
   box_truth_scaled.append(tmp3)
   object_type.append(tmp)
   objects_per_image[i]=icount
#   if(objects_per_image[i]<2):
#       print("this one",str(i))
#     plt.show(plt.imshow(pixel_array[i]))
# multiple planes 13, 14,33
#
# image 41 has 17 planes
#


#this was to do k-means using R
with open('boxes','w') as file:
  for i in range(0,nimages):
    jmax=len(box_truth_scaled[i])
    for j in range(0,jmax):
        dy=box_truth_scaled[i][j][1]-box_truth_scaled[i][j][0]
        dx=box_truth_scaled[i][j][3]-box_truth_scaled[i][j][2]
        file.write(str(dy)+","+str(dx)+"\n")

#R output for k-means(k=5)
#> out$centers
#          y         x
#1  38.76812  55.95652
#2 103.86232 152.39130
#3 319.44521 373.26712
#4 143.50413 265.31405
#5 183.04887 369.18797
bounds=[]
with open('bound_boxes') as file:
    bounds=file.readlines()
num_bound_boxes=len(bounds)
bound_box_hxw=np.zeros((num_bound_boxes,2))
for k in range(0,num_bound_boxes):
    tmp=bounds[k].split()
    bound_box_hxw[k,0]=float(tmp[0])
    bound_box_hxw[k,1]=float(tmp[1])


# let use a 15/15 grid
igrid_cells=10
iedge_size=inew/igrid_cells
cells_with_objects=[]
train_tensor=np.zeros((nimages,igrid_cells,igrid_cells,1))
for ipic in range(0,nimages):    
    nobjects=int(objects_per_image[ipic])
    tmp=[]
    for jobject in range(0,nobjects):
        ymin=box_truth_scaled[ipic][jobject][0]
        ymax=box_truth_scaled[ipic][jobject][1]
        xmin=box_truth_scaled[ipic][jobject][2]
        xmax=box_truth_scaled[ipic][jobject][3]
        ymid=ymin+(ymax-ymin)/2.0
        xmid=xmin+(xmax-xmin)/2.0
        icelly=int(np.floor(ymid/iedge_size))#+1
        icellx=int(np.floor(xmid/iedge_size))#+1
        print(ipic,nobjects,jobject,box_truth_scaled[ipic][jobject],ymid,xmid,icelly,icellx)
        tmp.append([icelly,icellx])
        train_tensor[ipic,icelly,icellx,0]=1 #probability of object
#        train_tensor[ipic,icelly,icellx,1]=1 #probability of plane in             
    cells_with_objects.append(tmp)    

#test viz
for ipic in range(0,nimages):
    #add grid lines
    for jcell in range(0,igrid_cells):
        iline=int(np.ceil((jcell+1)*iedge_size))-1
#        pixel_array[ipic,iline,0:inew,:]=0
#        pixel_array[ipic,0:inew,iline,:]=0
    #highlight grid cell assigned to object
    for k in range(0,int(objects_per_image[ipic])):
           ycell=cells_with_objects[ipic][k][0]
           xcell=cells_with_objects[ipic][k][1]
           ymax=int(np.ceil((ycell+1)*iedge_size))-1
           ymin=int(np.floor((ycell)*iedge_size))-1
           xmax=int(np.ceil((xcell+1)*iedge_size))-1
           xmin=int(np.floor((xcell)*iedge_size))-1
           xmid=int(xmin+(xmax-xmin)/2)
           ymid=int(ymin+(ymax-ymin)/2)
#           pixel_array[ipic,ymin:ymax,xmin:xmax,:]=0.
#           pixel_array[ipic,ymin:ymax,xmin:xmax,0]=1.
#           pixel_array[ipic,ymin:ymax,xmin:xmax,:]=0.
#           pixel_array[ipic,ymin:ymax,xmin:xmax,0]=1.
           pixel_array[ipic,(ymid-10):(ymid+10),(xmid-10):(xmid+10),:]=0.
           pixel_array[ipic,(ymid-10):(ymid+10),(xmid-10):(xmid+10),0]=1.
           for ibox in range(0,num_bound_boxes):
               xmax1=int(np.ceil(xmin+bound_box_hxw[ibox,1]/2))
               if(xmax1>inew):
                   xmax1=inew-1
               xmin1=int(np.ceil(xmin-bound_box_hxw[ibox,1]/2))
               if(xmin1<0):
                   xmin1=0
               ymax1=int(np.ceil(ymin+bound_box_hxw[ibox,0]/2))
               if(ymax1>inew):
                   ymax1=inew-1
               ymin1=int(np.ceil(ymin-bound_box_hxw[ibox,0]/2))
               if(ymin1<0):
                   ymin1=0
#               pixel_array[ipic,ymin1,xmin1:xmax1,:]=0
#               pixel_array[ipic,ymax1,xmin1:xmax1,:]=0
#               pixel_array[ipic,ymin1:ymax1,xmin1,:]=0
#               pixel_array[ipic,ymin1:ymax1,xmax1,:]=0


#    plt.show(plt.imshow(pixel_array[ipic]))



#CNN
np.random.seed(100)
model = keras.Sequential()
model.add(keras.layers.Conv2D(input_shape=(inew,inew,3),filters=8,kernel_size=2,strides=(1,1),padding='valid',data_format="channels_last",use_bias=True,activation=tf.nn.relu))
model.add(keras.layers.MaxPooling2D(pool_size=(3,3), strides=(3,3), padding='valid'))
#model.add(keras.layers.BatchNormalization(axis=1, epsilon=0.001))
model.add(keras.layers.Conv2D(filters=8,kernel_size=2,strides=(1,1),padding='valid',use_bias=True,activation=tf.nn.relu))
model.add(keras.layers.MaxPooling2D(pool_size=(3,3), strides=(3,3), padding='valid'))
#model.add(keras.layers.BatchNormalization(axis=1, epsilon=0.001))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(igrid_cells*igrid_cells, activation=tf.nn.relu))
model.add(keras.layers.Dense(igrid_cells*igrid_cells, activation='sigmoid'))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='mean_squared_error',
              metrics=['mse'])


np.random.seed(10)
xtrain, xtest, ytrain, ytest =train_test_split(pixel_array,train_tensor,test_size=.2)

#k=len(xtrain)
#yt=[]
#for i in range(0,k):
#    yt.append(train_tensor[i,:,:,:])

model.fit(xtrain,ytrain)
    



