import os
import numpy as np
from tqdm import tqdm
import keras.backend as K
import cv2 as cv
from keras.models import Model
from keras.layers import Input,BatchNormalization,Lambda
from keras.layers import  Dropout, BatchNormalization
from keras.layers.core import Dropout, Lambda
from keras.layers import Input, Conv3D, MaxPooling3D, Dropout, BatchNormalization
import tensorflow as tf
from volumentations import *
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate


INPUT_WIDTH=64
INPUT_HEIGHT=64
INPUT_DEPTH=64
INPUT_CHANNEL=1
OUTPUT_CHANNEL=1
TRAIN_SEG_LEARNING_RATE=0.001
SMOOTH = 1.0

def getWeightsPath(dirpath,choix_lr,choix1,choix2):
    list1=os.listdir(dirpath)
    list1.sort()
    for x in list1:
        path2=os.path.join(dirpath,x)
        if(choix_lr in x):
            for x1 in os.listdir(path2):
                if(choix1 in x1):
                    path3=os.path.join(path2,x1)
                    for x2 in os.listdir(path3):
                        if(choix2 in x2 and 'after' in x2): 
                            weights_segmentation_after=os.path.join(path3,x2)
                        if(choix2 in x2 and 'after' not in x2):
                            weights_segmentation_best=os.path.join(path3,x2)    
    return weights_segmentation_best,weights_segmentation_after

def findNoduleCoordinatesinBlock(mask,coords,plus=5):
    mask=mask.astype(dtype=np.uint8)
    list_coo1=[]
    list_coo2=[]
    for i in range(mask.shape[0]): 
        thresh = cv.threshold(mask[i],0,255,cv.THRESH_OTSU + cv.THRESH_BINARY)[1]
        cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#         if(len(cnts)>0):
#             print(i,len(cnts))
        for c in cnts:
            x,y,w,h = cv.boundingRect(c)
            x1=(x,y)
            x2=(x+w//2,y+h//2)
            r=max(w//2+1,h//2+1)
#             print(x,y,w,h)
#             print(x1,x2,r)
            coords1=(i,x2[1],x2[0],r)
            coords2=(coords[0]+coords1[0],coords[1]+coords1[1],coords[2]+coords1[2],r)
#             print(coords,coords1,coords2)
            list_coo1.append(coords1)
            list_coo2.append(coords2)
#             print()
    return  list_coo1,list_coo2       

def create3dBlock(img,coords,size_3d=64):

    cimg=np.full((size_3d,size_3d,size_3d),170,dtype=np.uint8)


    x=int(coords[2]-size_3d//2)
    y=int(coords[1]-size_3d//2)
    z=int(coords[0]-size_3d//2)
    zd=coords[0]
  

    ax=x
    bx=x+size_3d
    ay=y
    by=y+size_3d
    az=z
    bz=z+size_3d


    if(x+size_3d>img.shape[3]):
        ax=img.shape[3]-size_3d
        bx=img.shape[3]
    if(y+size_3d>img.shape[2]):
        ay=img.shape[2]-size_3d
        by=img.shape[2]
    if(z+size_3d>img.shape[1]):
        az=img.shape[1]-size_3d
        bz=img.shape[1]    
    if(x<0):
        ax=0
        bx=size_3d
    if(y<0):
        ay=0
        by=size_3d
    if(z<0):
        az=0
        bz=size_3d

    apx=0
    apy=0
    apz=0
    if(ax<0):
        apx=abs(ax)
        ax=0

    if(ay<0):
        apy=abs(ay)
        ay=0

    if(az<0):
        apz=abs(az)
        az=0

    start_coords=(az,ay,ax)
    cimg[0:size_3d-apz,0:size_3d-apy,0:size_3d-apx]=img[0][az:bz,ay:by,ax:bx]

    
    return cimg,start_coords 

def getAll3dBlocks(img,sp=[32,64,64]):
    list_coords=[]
    for z in range(0,img.shape[1],sp[0]):
        zf=z+64
        if(zf>img.shape[1]):
            zf=img.shape[1] 
            z=zf-64
        for y in range(0,img.shape[2],sp[1]):
            yf=y+64
            if(yf>img.shape[2]):
                yf=img.shape[2]
                y=yf-64
            for x in range(0,img.shape[3],sp[2]):
                xf=x+64
                if(xf>img.shape[3]):
                    xf=img.shape[3]
                    x=xf-64
                #block=img[0][z:z+64,y:y+64,x:x+64]
                list_coords.append((z,y,x))

    return list_coords 

def Load3dBlock(rimg):
    rgbimg = np.zeros((rimg.shape[0],rimg.shape[1],rimg.shape[2],3),dtype=rimg.dtype)

    
    for z in range(rimg.shape[0]):
            rgbimg[z]=cv.cvtColor(rimg[z],cv.COLOR_GRAY2RGB)
   
    rgbimg=rgbimg.astype(dtype=np.uint8)
     
       
    return rgbimg

def drawAllResults(img,list_coords,color):
    img2=np.copy(img)
    for i in tqdm(range(len(list_coords))):
        for c1 in list_coords[i]:
            img2[c1[0]] = cv2.circle(img2[c1[0]], (c1[2],c1[1]), c1[3], color, 1)
    return img2                                         

def show_predictions2(model,sample_image,coords,show=True):


    #prediction
    sample_image=sample_image.astype(np.float32)
    pred_mask = model.predict(sample_image[tf.newaxis, ...])
    predmask=(pred_mask[0]*255).astype(dtype=np.uint8)

    sample_image = sample_image[..., np.newaxis]
    sample_image=sample_image.astype(np.uint8)

    l1,l2=findNoduleCoordinatesinBlock(predmask,coords)
    
    return  np.max(np.unique(pred_mask.astype(dtype=np.uint8))),l1,l2

def getNodulesCoordinates3(model,img):
    

    img2=np.copy(img)

    list_coords=getAll3dBlocks(img)


    results_list_coords=[]

    print(len(list_coords))
    for id,c in tqdm(enumerate(list_coords)):
        
        coords=list_coords[id]
        sample_image=img[0][coords[0]:coords[0]+64,coords[1]:coords[1]+64,coords[2]:coords[2]+64]


        predval,l1,l2=show_predictions2(model,sample_image,coords,show=False)

   
        if(predval>0):
                            
            #create nodule centered block
            ind=len(l2)//2
            coo=(l2[ind][0],l2[ind][1],l2[ind][2])
            block2,st=create3dBlock(img,coo) 
            
            predval2,l3,l4=show_predictions2(model,block2,st,show=False)
            if(predval2>0):

                #get block
                block3=img[0][st[0]:st[0]+64,st[1]:st[1]+64,st[2]:st[2]+64]
                final_block=Load3dBlock(block3)

                
                results_list_coords.append(l4)
    return results_list_coords


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def metrics_true_sum(y_true, y_pred):
    return K.sum(y_true)

def metrics_pred_sum(y_true, y_pred):
    return K.sum(y_pred)

def metrics_pred_max(y_true, y_pred):
    return K.max(y_pred)

def metrics_pred_min(y_true, y_pred):
    return K.min(y_pred)

def metrics_pred_mean(y_true, y_pred):
    return K.mean(y_pred)

def get_unet():
    inputs = Input((INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH, INPUT_CHANNEL))
    s = Lambda(lambda x: x / 255) (inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(s)
    conv1 = Dropout(0.1) (conv1)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    pool1 = Dropout(0.5) (pool1)
    pool1 = BatchNormalization()(pool1)
   
    conv2 = Conv3D(64, (3, 3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = Dropout(0.1) (conv2)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    pool2 = Dropout(0.5) (pool2)
    pool2 = BatchNormalization()(pool2)
    
    conv3 = Conv3D(128, (3, 3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = Dropout(0.2) (conv3)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    pool3 = Dropout(0.2) (pool3)
    pool3 = BatchNormalization()(pool3)

    
    
    
    conv4 = Conv3D(256, (3, 3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = Dropout(0.3) (conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(conv4)

    
    
    
    
    
    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv3], axis=-1)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(up7)
    conv7 = Dropout(0.2) (conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(conv7)

    up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2], axis=-1)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(up8)
    conv8 = Dropout(0.2) (conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(conv8)

    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(up9)
    conv9 = Dropout(0.1) (conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(conv9)

    conv10 = Conv3D(OUTPUT_CHANNEL, (1, 1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=TRAIN_SEG_LEARNING_RATE), loss=dice_coef_loss,
                  metrics=[dice_coef])

    return model

