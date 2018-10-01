#change all double precision models to float and use
#model().float() to change all NN weights to floats

import torch
import torch.nn as nn
import torch.utils.data as data
import time
import matplotlib.pyplot as plt
import math
import sys
import scipy
import numpy as np
import random
import os
#import pandas as pd
#from pandas import DataFrame
#from numpy import linalg as la
#from sklearn.neural_network import MLPRegressor
#import tensorflow as tf
#from tensorflow import keras
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Dropout
#from keras.optimizers import SGD, Adam, Nadam
from skimage import io
from skimage.transform import resize
from sklearn.cross_validation import train_test_split
#from sklearn.metrics import confusion_matrix
#from sklearn.preprocessing import StandardScaler
import xml.etree.ElementTree as ET
#import progressbar
from time import time


device="cpu"
if(torch.cuda.is_available()):
   device = torch.device("cuda:0")

for i in range(0,10):
   print("Running on "+str(device))

ngpu = torch.cuda.device_count()
print("number of gpus "+str(ngpu))

#torch.cuda.seed(100)



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
#nimages=int(sys.argv[1])
nimages=50
inew=200
pixel_array=np.zeros((nimages,inew,inew,3))
objects_per_image=np.zeros((nimages))
object_type=[]
box_truth=[]
box_truth_scaled=[]
original_size=np.zeros((nimages,3))
picture_files=[]


#for i in progressbar.progressbar(range(0,nimages)):
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
#        pixel_array[i,dylow,dxlow:dxhigh,:]=0
#        pixel_array[i,dylow:dyhigh,dxlow,:]=0
#        pixel_array[i,dyhigh,dxlow:dxhigh,:]=0
#        pixel_array[i,dylow:dyhigh,dxhigh,:]=0
        tmp3.append([dylow,dyhigh,dxlow,dxhigh])
   box_truth.append(tmp2)
   box_truth_scaled.append(tmp3)
   object_type.append(tmp)
   objects_per_image[i]=icount
#   if(objects_per_image[i]<2):
#       print("this one",str(i))
#   plt.show(plt.imshow(pixel_array[i]))
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
cells_with_objects_row=[]
train_tensor=np.zeros((nimages,igrid_cells,igrid_cells,5))
#cell_weights = np.zeros((nimages,igrid_cells,igrid_cells,5))
for ipic in range(0,nimages):
    nobjects=int(objects_per_image[ipic])
    tmp=[]
    tmp2=[]
    tmp3=[]
    for jobject in range(0,nobjects):
        ymin=box_truth_scaled[ipic][jobject][0]
        ymax=box_truth_scaled[ipic][jobject][1]
        xmin=box_truth_scaled[ipic][jobject][2]
        xmax=box_truth_scaled[ipic][jobject][3]
        ymid=ymin+(ymax-ymin)/2.0
        xmid=xmin+(xmax-xmin)/2.0
        icelly=int(np.floor(ymid/iedge_size))#+1
        icellx=int(np.floor(xmid/iedge_size))#+1
        yoff=ymid/iedge_size - icelly
        xoff=xmid/iedge_size - icellx
        tmp.append([icelly,icellx])
        tmp2.append(icelly*igrid_cells+icellx)
#        tmp3.append([yoff,xoff])
        train_tensor[ipic,icelly,icellx,0]=1 
        train_tensor[ipic,icelly,icellx,1]=yoff
        train_tensor[ipic,icelly,icellx,2]=xoff
        train_tensor[ipic,icelly,icellx,3]=(ymax-ymin)/inew
        train_tensor[ipic,icelly,icellx,4]=(xmax-xmin)/inew
#        train_tensor[ipic,icelly,icellx,1]=1 #probability of plane in
    cells_with_objects.append(tmp)
    cells_with_objects_row.append(tmp2)

map_row_to_cell=[]
for i in range(0,igrid_cells):
    for j in range(0,igrid_cells):
        map_row_to_cell.append([i,j])
    
#scale_value=1.
#cell_weights[train_tensor[:,:,:,0]==1]=scale_value
#scale_value2=.1
#cell_weights[train_tensor[:,:,:,0]==0]=scale_value2

cell_weights = np.zeros((nimages,igrid_cells,igrid_cells,5))
bb_weights =   np.zeros((nimages,igrid_cells,igrid_cells,5))
scale_value=1.0
scale_value2=0.1
scale_value3=1.0
for ipic in range(0,nimages):
   for jcell in range(0,igrid_cells):
       for kcell in range(0,igrid_cells):
          if(train_tensor[ipic,jcell,kcell,0]==1):
              cell_weights[ipic,jcell,kcell,0]=scale_value
              bb_weights[ipic,jcell,kcell,1:]=scale_value3
          else:
              cell_weights[ipic,jcell,kcell,0]=scale_value2





'''
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
           ymax=int(np.ceil((ycell+1)*iedge_size))
           ymin=int(np.floor((ycell)*iedge_size))
           xmax=int(np.ceil((xcell+1)*iedge_size))
           xmin=int(np.floor((xcell)*iedge_size))
           xmid=int(xmin+(xmax-xmin)/2)
           ymid=int(ymin+(ymax-ymin)/2)
           pixel_array[ipic,ymin:ymax,xmin:xmax,:]=0.
           pixel_array[ipic,ymin:ymax,xmin:xmax,0]=1.
           ycenter=ycell*iedge_size+iedge_size*train_tensor[ipic,ycell,xcell,1]
           xcenter=xcell*iedge_size+iedge_size*train_tensor[ipic,ycell,xcell,2]
#           print(str(train_tensor[ipic,ycell,xcell,1])+" "+str(train_tensor[ipic,ycell,xcell,2]))
           ylow=int(ycenter-train_tensor[ipic,ycell,xcell,3]/2)
           yhigh=int(ycenter+train_tensor[ipic,ycell,xcell,3]/2)
           xlow=int(xcenter-train_tensor[ipic,ycell,xcell,4]/2)
           xhigh=int(xcenter+train_tensor[ipic,ycell,xcell,4]/2)
           pixel_array[ipic,ylow,xlow:xhigh,:]=0
           pixel_array[ipic,ylow:yhigh,xlow,:]=0
           pixel_array[ipic,yhigh,xlow:xhigh,:]=0
           pixel_array[ipic,ylow:yhigh,xhigh,:]=0
           bx=int(xcenter)
           by=int(ycenter)
           pixel_array[ipic,(by-1):(by+1),(bx-1):(bx+1),:]=0

           

#           for ibox in range(0,num_bound_boxes):
#               xmax1=int(np.ceil(xmin+bound_box_hxw[ibox,1]/2))
#               if(xmax1>inew):
#                   xmax1=inew-1
#               xmin1=int(np.ceil(xmin-bound_box_hxw[ibox,1]/2))
#               if(xmin1<0):
#                   xmin1=0
#               ymax1=int(np.ceil(ymin+bound_box_hxw[ibox,0]/2))
#               if(ymax1>inew):
#                   ymax1=inew-1
#               ymin1=int(np.ceil(ymin-bound_box_hxw[ibox,0]/2))
#               if(ymin1<0):
#                   ymin1=0
#               pixel_array[ipic,ymin1,xmin1:xmax1,:]=0
#               pixel_array[ipic,ymax1,xmin1:xmax1,:]=0
#               pixel_array[ipic,ymin1:ymax1,xmin1,:]=0
#               pixel_array[ipic,ymin1:ymax1,xmax1,:]=0


#    plt.show(plt.imshow(pixel_array[ipic]))
'''



np.random.seed(10)
indices=np.arange(nimages)
xtrain, xtest, ytrain, ytest,indtrain,indtest,wtrain,wtest,btrain,btest =train_test_split(pixel_array,train_tensor,indices,cell_weights,bb_weights,test_size=.1)
m=len(ytrain)
tmp=[]
tmp2=[]
tmp3=[]
for i in range(0,m):
    tmp.append(ytrain[i].flatten(order='C'))
    tmp2.append(wtrain[i].flatten(order='C'))
    tmp3.append(btrain[i].flatten(order='C'))
ytrain2=np.asarray(tmp)
wtrain2=np.asarray(tmp2)
btrain2=np.asarray(tmp3)

npixels=inew
scale_values=np.zeros((npixels,npixels,3,2))
i=0
#while i<npixels:
#for i in progressbar.progressbar(range(0,npixels)):
for i in range(0,npixels):
    j=0
    while j<npixels:
        k=0
        while k<3:
            scale_values[i,j,k,0]=np.mean(xtrain[:,i,j,k])
            scale_values[i,j,k,1]=np.std(xtrain[:,i,j,k]  )
            xtrain[:,i,j,k]=xtrain[:,i,j,k]-scale_values[i,j,k,0]
            xtest[:,i,j,k]=xtest[:,i,j,k]-scale_values[i,j,k,0]
            if(scale_values[i,j,k,1] >0.0):
               xtrain[:,i,j,k]=xtrain[:,i,j,k]/scale_values[i,j,k,1]
               xtest[:,i,j,k]=xtest[:,i,j,k]/scale_values[i,j,k,1]
            k+=1
        j+=1
#    i+=1

#CNN
np.random.seed(100)

class NeuralNet(nn.Module):

    def __init__(self):
       super(NeuralNet, self).__init__()
       self.conv1 = nn.Conv2d(3, 64, (3 , 3), padding=0) #198x198x64 formula is output_width=(pic_width - kernel_size +2Pads)/stride + 1
       self.bnrm1 = nn.BatchNorm2d(64)
       self.act1 = nn.Tanh()

       self.conv2 = nn.Conv2d(64, 64, (3 , 3), padding=0) #196x196x64
       self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)) #98x98x64
       self.bnrm2 = nn.BatchNorm2d(64)
       self.act2 = nn.Tanh()


       self.conv3 = nn.Conv2d(64, 128, (3 , 3), padding=0) #96x96x128
       self.pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))#48x48x128
       self.bnrm3 = nn.BatchNorm2d(128)
       self.act3 = nn.Tanh()
       
       self.conv4 = nn.Conv2d(128, 256, (3 , 3), padding=0) # 46x46x256
       self.pool4 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)) #23x23x256
       self.bnrm4 = nn.BatchNorm2d(256)
       self.act4 = nn.Tanh()

       self.conv5 = nn.Conv2d(256, 512, (3 , 3)) #21x21x512
       self.pool5 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)) 
#       self.bnrm5 = nn.BatchNorm2d(512)
       self.act5 = nn.Tanh()

       self.fc6 = nn.Linear(51200,1024)
       self.bn6 = nn.BatchNorm1d(51200)
       self.act6 = nn.Tanh()
       self.fc7 = nn.Linear(1024,500)
       self.fc8 = nn.Sigmoid()

    def forward(self, x):
       out = self.conv1(x)
       out = self.bnrm1(out)
       out = self.act1(out)
       out = self.conv2(out)
       out = self.bnrm2(out)
       out = self.act2(out)
       out = self.pool2(out)
       out = self.conv3(out)
       out = self.bnrm3(out)
       out = self.act3(out)
       out = self.pool3(out)
       out = self.conv4(out)
       out = self.bnrm4(out)
       out = self.act4(out)
       out = self.pool4(out)
       out = self.conv5(out)
       out = self.act5(out)
       out = self.pool5(out)
#       out = self.bnrm5(out)
       out = out.view(-1, self.num_flat_features(out))
       out = self.bn6(out)
       out = self.fc6(out)
       out = self.act6(out)
       out = self.fc7(out)
       yout = self.fc8(out)
       return yout

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features




#xtrain_tch=torch.from_numpy(xtrain).float()
model=NeuralNet().float().to(device)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        print("params in layer "+str(nn))
        pp += nn
    return pp

nparams=get_n_params(model)
print("Number of parameters is "+str(nparams))

print("start fit")
#pred=model(xtrain_tch.permute(0,3 , 1, 2))


cutoff=.000001
xtrain_tch=torch.from_numpy(xtrain).float()
ytrain_tch=torch.from_numpy(ytrain2).float()
indices=torch.from_numpy(np.arange(len(xtrain_tch))).float()
metric=nn.MSELoss()
wts_torch=torch.from_numpy(wtrain2).float()
wts_torch2=torch.from_numpy(btrain2).float()
t=data.TensorDataset(xtrain_tch,ytrain_tch,indices,wts_torch,wts_torch2)
batch_size=20
batches=data.DataLoader(t,batch_size=batch_size,shuffle=True)
mets=np.zeros((len(batches),1))
mets2=np.zeros((len(batches),1))
xtest_tch=torch.from_numpy(xtest).float()
ytest_tch=torch.from_numpy(ytest).float()
tmp=[]
optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs=0
history=np.zeros((num_epochs,1))
t1=time()
model=torch.load("torchfitboxes.dat",map_location="cpu").float()
print(optimizer)
best_one=1000.0
best_raw=1000.0
jbest=0
for epoch in range(0,num_epochs):
     with open("epoch","w") as file:
         file.write(str("current epoch: ")+str(epoch)+"\n")
         file.write("Best so far: "+str(best_one)+" at epoch "+str(jbest)+"\n")
         file.write("Best RAW so far: "+str(best_raw))
     file.close()
     icount=-1
     tstart=time()
     for batch in batches:
#x.permute(2, 0, 1).size()
        these_wts=batch[3].float().to(device)
        these_wts2=batch[4].float().to(device)
        refvals=batch[1].float().to(device)
        pred = model(batch[0].permute(0,3 , 1, 2).float().to(device))
#        metval = metric(pred,refvals).item()  
#        loss2 = metric(pred, batch[1].float())
#        loss=torch.mean(these_wts*((pred-refvals)**2)) #this is weight*squared erros
#        loss+=torch.mean(these_wts2*((pred-refvals)**2))
        loss=0
        loss_raw=0
        num_terms=0
        k=len(batch[0])
        for iphoto in range(0,k):
           pred_sq=pred[iphoto].view(igrid_cells,igrid_cells,5)
           ref_sq=refvals[iphoto].view(igrid_cells,igrid_cells,5)
           w1=these_wts[iphoto].view(igrid_cells,igrid_cells,5)
           w2=these_wts2[iphoto].view(igrid_cells,igrid_cells,5)
           for icell in range(0,igrid_cells):
               for jcell in range(0,igrid_cells):
                  loss+=w1[icell,jcell,0]*(pred_sq[icell,jcell,0]-ref_sq[icell,jcell,0])**2
                  loss_raw+=(pred_sq[icell,jcell,0]-ref_sq[icell,jcell,0])**2
                  num_terms+=1
                  if(ref_sq[icell,jcell,0]==1):
                     for krow in range(1,5):
                        loss+=w2[icell,jcell,krow]*(pred_sq[icell,jcell,krow]-ref_sq[icell,jcell,krow])**2
                        loss_raw+=(pred_sq[icell,jcell,krow]-ref_sq[icell,jcell,krow])**2
                        num_terms+=1
# Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        icount+=1
        mets[icount,0]=loss.item()/num_terms
        mets2[icount,0]=loss_raw.item()/num_terms
#        print(str(icount)+" "+str(loss.item()))
     eloss=np.mean(mets)
     eloss_raw=np.mean(mets2)
     tend=time()
     if(eloss < best_one):
        print("New best fit achieved")
        jbest=epoch
        best_one = eloss
        best_raw = eloss_raw
        torch.save(model,"torchfitboxes.dat")
     print('mean loss after epoch '+str(epoch)+': '+str(eloss))
     print("Best so far: "+str(best_one)+" at epoch "+str(jbest))
     print("Best RAW so far: "+str(best_raw))
     print("Metric (Raw MSE) "+str(eloss_raw))
     print("Epoch Time(s): "+str(tend-tstart))
     print("---------------------------------------")
     history[epoch][0]=eloss
#     pred=model(xtest_tch.permute(0,3 , 1, 2))
#     imax=len(test_images)
     i=0
     if(eloss<cutoff):
         break
print("")
print("")
print(" ______      ___   ____  _____  ________ ") 
print("|_   _ `.  .'   `.|_   \|_   _||_   __  | ")
print("  | | `. \/  .-.  \ |   \ | |    | |_ \_| ")
print("  | |  | || |   | | | |\ \| |    |  _| _  ")
print(" _| |_.' /\  `-'  /_| |_\   |_  _| |__/ | ")
print("|______.'  `.___.'|_____|\____||________| ")
                                          
print("Best value was "+str(best_one)+" at epoch "+str(jbest)) 
print("Best raw MSE was "+str(best_raw))
t2=time()
print("total time was "+str(t2-t1)+" seconds")
#torch.save(model,"torchfitboxes.dat")








#m=len(ytest2)
#tmp=[]
#for i in range(0,m):
#    tmp.append(ytest[i].flatten(order='C'))
#ytest2=np.asarray(tmp)
model.eval()
k=len(xtrain_tch)
box_tol=0.4
for ipic in range(0,k):
    pred = model(xtrain_tch[ipic:ipic+1].permute(0,3 , 1, 2))#.to(device))
    pred_sq=pred.view(igrid_cells,igrid_cells,5).detach().numpy()
    pred3=pred_sq[:,:,0].flatten(order='C')
    pred2=np.argwhere(pred3>box_tol)
    imap=indtrain[ipic]
    num_boxes=int(objects_per_image[imap])
    num_predicted=len(pred2)
    this_pic=pixel_array[indtrain[ipic]]
    print("****************************")
    print("# objects truth: ",str(num_boxes))
    print("# objects found: ",str(num_predicted))
    print("train index: ",str(ipic))
    print("original index: ",str(imap))
#    for jcell in range(0,igrid_cells):
#              iline=int(np.ceil((jcell+1)*iedge_size))-1
#              this_pic[iline,0:inew,:]=0
#              this_pic[0:inew,iline,:]=0
    for jbox in range(0,num_boxes):
        ymax=box_truth_scaled[imap][jbox][1]
        ymin=box_truth_scaled[imap][jbox][0]
        xmax=box_truth_scaled[imap][jbox][3]
        xmin=box_truth_scaled[imap][jbox][2]
        this_pic[ymin:ymax,xmin,:]=0.
        this_pic[ymin:ymax,xmax,:]=0.
        this_pic[ymin,xmin:xmax,:]=0.
        this_pic[ymax,xmin:xmax,:]=0.
        this_pic[ymin:ymax,xmin,1]=1.
        this_pic[ymin:ymax,xmax,1]=1.
        this_pic[ymin,xmin:xmax,1]=1.
        this_pic[ymax,xmin:xmax,1]=1.
        

    tmp=[]
    for kbox in range(0,num_predicted):
        detector_location=map_row_to_cell[int(pred2[kbox])]
#        print(detector_location)
        ycell=detector_location[0]
        xcell=detector_location[1]
        ymin=int(np.floor((ycell)*iedge_size))
        xmin=int(np.floor((xcell)*iedge_size))

        ycenter=ycell*iedge_size+iedge_size*pred_sq[ycell,xcell,1]
        xcenter=xcell*iedge_size+iedge_size*pred_sq[ycell,xcell,2]
        ylow=int(ycenter-inew*pred_sq[ycell,xcell,3]/2)
        if(ylow<0):
           ylow=0
        yhigh=int(ycenter+inew*pred_sq[ycell,xcell,3]/2)
        if(yhigh>inew-1):
           yhigh=inew-1
        xlow=int(xcenter-inew*pred_sq[ycell,xcell,4]/2)
        if(xlow<0):
           xlow=0
        xhigh=int(xcenter+inew*pred_sq[ycell,xcell,4]/2)
        if(xhigh>inew-1):
           xhigh=inew-1
#        tmp.append([ymin,xmin,pred[0][pred2[kbox][1]].item()])
#        this_pic[ymin:ymax,xmin,:]=0.
#        this_pic[ymin:ymax,xmax,:]=0.
#        this_pic[ymin,xmin:xmax,:]=0.
#        this_pic[ymax,xmin:xmax,:]=0.
        this_pic[ylow:yhigh,xlow,:]=0.
        this_pic[ylow:yhigh,xhigh,:]=0.
        this_pic[ylow,xlow:xhigh,:]=0.
        this_pic[yhigh,xlow:xhigh,:]=0.
        this_pic[ylow:yhigh,xlow,0]=1.
        this_pic[ylow:yhigh,xhigh,0]=1.
        this_pic[ylow,xlow:xhigh,0]=1.
        this_pic[yhigh,xlow:xhigh,0]=1.


        
#        this_pic[ymin:ymax,xmin:xmax,1]=1.
#        this_pic[ymin,xmin:xmin:xmax,1]=1.
#        this_pic[ymax,xmin:xmin:xmax,1]=1.

    plt.subplot(1,2,1)
    plt.imshow(this_pic)
#    for kbox in range(0,num_predicted):
#        xpt=tmp[kbox][1]
#        ypt=tmp[kbox][0]
#        val=tmp[kbox][2]
#        plt.text(xpt,ypt,str(val)[0:6])
    plt.subplot(1,2,2)
    original=io.imread(picture_files[indtrain[ipic]])
    plt.imshow(original)
#    plt.show()
    plt.pause(2)
    plt.close()

    #        pixel_array[ipic,(ymid-10):(ymid+10),(xmid-10):(xmid+10),:]=0.
#        pixel_array[ipic,(ymid-10):(ymid+10),(xmid-10):(xmid+10),0]=1.



    



