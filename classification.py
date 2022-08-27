import os
import numpy as np
import cv2 as cv
from keras.models import Sequential
from keras.layers import Dense, Flatten
import tensorflow as tf
from classification_models_3D.tfkeras import Classifiers
from volumentations import *
import nashpy as nash
TRAIN_CLASSIFY_USE_BN=True
TRAIN_CLASSIFY_LEARNING_RATE=1e-4
CLASSIFY_INPUT_WIDTH=64
CLASSIFY_INPUT_HEIGHT=64
CLASSIFY_INPUT_DEPTH=64
CLASSIFY_INPUT_CHANNEL=1


def getClassification(list_coords,img):
    dirpath2='Classification/classification weights'
    weights_classification=getWeightsPath(md='DenseNet121',nb='1',after_best='after',dirpath2=dirpath2)
    print(weights_classification)
    model_densenet121,_=get_densenet121_model()
    model_densenet121.load_weights(weights_classification) 
    print("Loaded model from disk")

    weights_classification2=getWeightsPath(md='ResNet50',nb='1',after_best='after',dirpath2=dirpath2)
    print(weights_classification2)
    model_resnet50,_=get_resnet50_model()
    model_resnet50.load_weights(weights_classification2) 
    print("Loaded model from disk")

    weights_classification3=getWeightsPath(md='InceptionV3',nb='1',after_best='after',dirpath2=dirpath2)
    print(weights_classification3)
    model_inceptionv3,_=get_inceptionv3_model()
    model_inceptionv3.load_weights(weights_classification3) 
    print("Loaded model from disk") 

    weights_classification4=getWeightsPath(md='InceptionResNetV2',nb='1',after_best='after',dirpath2=dirpath2)
    print(weights_classification4)
    model_inceptionresnetv2,_=get_inceptionresnetv2_model()
    model_inceptionresnetv2.load_weights(weights_classification4) 
    print("Loaded model from disk")

    all_models_results=getAllClassifications(list_coords,img,model_densenet121,model_resnet50,model_inceptionv3,model_inceptionresnetv2)

    return all_models_results

def get_resnet50_model():
    ResNet50, preprocess_input = Classifiers.get('resnet50')
    resnet_model = Sequential()
    pretrained_model = ResNet50(input_shape=(64, 64, 64, 3), weights='imagenet')

    resnet_model.add(pretrained_model)
    
    resnet_model.add(Flatten())
    resnet_model.add(Dense(512, activation='relu'))
    resnet_model.add(Dense(2, activation='softmax'))
    resnet_model.compile(tf.keras.optimizers.Adam(learning_rate=TRAIN_CLASSIFY_LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return resnet_model,preprocess_input

def get_densenet121_model():
    DenseNet121, preprocess_input = Classifiers.get('densenet121')
    model = Sequential()
    pretrained_model = DenseNet121(input_shape=(64, 64, 64, 3))


    model.add(pretrained_model)
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(tf.keras.optimizers.Adam(learning_rate=TRAIN_CLASSIFY_LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model,preprocess_input

def get_inceptionv3_model():
    InceptionV3, preprocess_input = Classifiers.get('inceptionv3')
    model = Sequential()
    pretrained_model = InceptionV3(input_shape=(64, 64, 64, 3),weights='imagenet')

    model.add(pretrained_model)
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(tf.keras.optimizers.Adam(learning_rate=TRAIN_CLASSIFY_LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model,preprocess_input

def get_inceptionresnetv2_model():
    InceptionResNetV2, preprocess_input = Classifiers.get('inceptionresnetv2')
    model = Sequential()
    pretrained_model = InceptionResNetV2(input_shape=(64, 64, 64, 3),weights='imagenet')


    model.add(pretrained_model)
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(tf.keras.optimizers.Adam(learning_rate=TRAIN_CLASSIFY_LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model,preprocess_input

def getWeightsPath(md,nb,after_best,dirpath2):
    list1=os.listdir(dirpath2)
    list1.sort()
    for x in list1:
        if(x==md):
            path2=os.path.join(dirpath2,x)
            #print(path2)
            for x1 in os.listdir(path2):
                if('.zip' not in x1):
                    print(x1)
                    if(after_best=='after'):
                        if(after_best in x1 and nb in x1):
                            weights_classification=os.path.join(path2,x1)
                            #print(weights_classification)   
                    else:
                        if('after' not in x1 and nb in x1):
                            weights_classification=os.path.join(path2,x1)
                            #print(weights_classification)    
    return weights_classification

def get3dBlock(img,cm):
    zm0=cm[0]
    rm=64//2
    zm=cm[0]-rm
    xm=cm[2]-rm
    ym=cm[1]-rm
    if(zm<0):
        zm=0
        zmp=64
    else:
        zmp=zm+64
        if(zmp>img.shape[1]):
            zmp=img.shape[1]
            zm=zmp-64

    if(ym<0):
        ym=0
        ymp=64
    else:
        ymp=ym+64
        if(ymp>img.shape[2]):
            ymp=img.shape[2]
            ym=ymp-64

    if(xm<0):
        xm=0
        xmp=64
    else:
        xmp=xm+64
        if(xmp>img.shape[3]):
            xmp=img.shape[3]
            xm=xmp-64

    return (zm,ym,xm)

def predict(model0,img,coord):
    coords=get3dBlock(img,coord)
    z=coords[0]
    y=coords[1]
    x=coords[2]
    #print(coords)
    block=img[0][z:z+64,y:y+64,x:x+64]
    rgb_block=Load3dBlock(block)
    X=np.zeros((1,64,64,64,3),dtype=np.uint8)
    X[0]=rgb_block
    #plot3dBlock(block)
    y_predicted= model0.predict(X)
    y_predicted2=np.argmax(y_predicted,1)
    y_predicted3=[np.round(y_predicted[0][0],4),np.round(y_predicted[0][1],4)]
    return y_predicted3,y_predicted2[0]

def Load3dBlock(rimg):
    rgbimg = np.zeros((rimg.shape[0],rimg.shape[1],rimg.shape[2],3),dtype=rimg.dtype)

    
    for z in range(rimg.shape[0]):
            rgbimg[z]=cv.cvtColor(rimg[z],cv.COLOR_GRAY2RGB)
   
    rgbimg=rgbimg.astype(dtype=np.uint8)
     
       
    return rgbimg

def printGame(A,B,rownames,colnames):
    print("***** Payoffs Matrix : *************************")
    d = {rownames[0]: [(A[0][0],B[0][0]),(A[0][1],B[0][1])],
    rownames[1]: [(A[1][0],B[1][0]),(A[1][1],B[1][1])]
    }
    print ("{:<20} {:<20} {:<20}".format('',colnames[0],colnames[1]))
    print()
    for k, v in d.items():
        print ("{:<20} {:<20} {:<20} ".format(k,str(v[0]),str(v[1])))
        print()
    print("************************************************")    

def GetFinalResultsWithThj(elm,thresh=0.5):
    ApplyThj=True
    game_values=set()
    #case:malignant
    cpt=0
    for c in elm:
        if(c[3]>=thresh):
            cpt+=1
    if(cpt==len(elm)):
        ApplyThj=False


    #case:benign
    cpt=0
    for c in elm:
        if(c[2]>=thresh):
            cpt+=1     
    if(cpt==len(elm)):
        ApplyThj=False


    # appliying Thj To Get Final Result if there was at least 2 stratigies for each player
    if(ApplyThj):  
        elm_m=[]
        elm_b=[]
        
        ms=[]
        bs=[]
        rn=[]
        cn=[]
        avg_b=0
        avg_m=0
        for c in elm:
            if(c[1]==1):
                avg_m+=0.5*c[3]
                elm_m.append(c)
            else:
                avg_b+=0.5*c[2]
                elm_b.append(c)
                
        if(len(elm_m)==len(elm_b)):
            for m in elm_m:
                rn.append(m[0])
                bs2=[]
                ms2=[]
                for b in elm_b:
                    gain_mal=m[3]
                    gain_be=b[2]
                    gain=gain_mal-gain_be
                    gain=np.round(gain,2)
                    #sts names
                    if(b[0] not in cn):
                        cn.append(b[0])
                    #add sts
                    ms2.append(gain)
                    bs2.append(-1*gain)
                bs.append(bs2)
                ms.append(ms2)


            Malignant_stratigies = np.array(ms) # A is the row player
            Benign_stratigies = np.array(bs) # B is the column player

            game1 = nash.Game(Malignant_stratigies,Benign_stratigies)
            final_values=getGameValueFromNashEquilibrium(game1,rn,cn)

            for j,val in enumerate(final_values):
                if(val[0]>0):
                    vf=1
                    avge=avg_m
                elif(val[0]<0):
                    vf=0
                    avge=avg_b
                else:
                    if(avg_m>=avg_b):
                        vf=2
                        avge=avg_m
                    else:
                        vf=-2
                        avge=avg_b
                        

                game_values.add((val[0],avge))
    return game_values    

def getGameValueFromNashEquilibrium(game,rn,cn):
    equilibria = game.support_enumeration()
    list_final_values=[]
    for i,eq in enumerate(equilibria):
        ind=(np.argmax(eq[0]),np.argmax(eq[1]))
        result=(rn[ind[0]],cn[ind[1]])
        A=[]
        B=[]
        for j in range(len(eq[0])):
             A.append(eq[0][j])
             B.append(eq[1][j])
        sigma_r = np.array(A)
        sigma_c = np.array(B)
        value=game[sigma_r, sigma_c]
        list_final_values.append(value)
    return list_final_values

def getAllClassifications(list_coords,img,model_densenet121,model_resnet50,model_inceptionv3,model_inceptionresnetv2):
    list_preds_per_coords=[]
    for c in list_coords:
        ind=len(c)//2
        test_c=(c[ind][0],c[ind][1],c[ind][2])
        nbb=0
        nbm=0
        elm=[]
        pred={}
        y,y2=predict(model_densenet121,img,test_c)
        elm.append(("DenseNet121",y2,y[0],y[1]))
        if(y2==0): 
            nbb+=1 
        else: 
            nbm+=1
        pred["DenseNet121"]=(y2,y)
        y,y2=predict(model_resnet50,img,test_c)
        elm.append(("ResNet50",y2,y[0],y[1]))
        if(y2==0): 
            nbb+=1 
        else: 
            nbm+=1
        pred["ResNet50"]=(y2,y)
        y,y2=predict(model_inceptionv3,img,test_c)
        elm.append(("InceptionV3",y2,y[0],y[1]))
        if(y2==0): 
            nbb+=1 
        else: 
            nbm+=1
        pred["InceptionV3"]=(y2,y)
        y,y2=predict(model_inceptionresnetv2,img,test_c)
        elm.append(("InceptionResNetV2",y2,y[0],y[1]))
        if(y2==0): 
            nbb+=1 
        else: 
            nbm+=1
        pred["InceptionResNetV2"]=(y2,y)
        res=GetFinalResultsWithThj(elm)
        pred["THJ"]=res
        #print(nbm,nbb)
        
        if(nbb!=nbm):
            if(nbb>nbm):
                v=max( pred["DenseNet121"][1][0] ,pred["ResNet50"][1][0],pred["InceptionV3"][1][0],pred["InceptionResNetV2"][1][0] )
                final_prediction=(0.0, v)
            else:
                v=max( pred["DenseNet121"][1][1] ,pred["ResNet50"][1][1],pred["InceptionV3"][1][1],pred["InceptionResNetV2"][1][1] )
                final_prediction=(1.0 ,v)
        else:
            lres=list(res)
            final_prediction=lres[0]
            print(lres[0])            
        pred["FinalPrediction"]=final_prediction
        list_preds_per_coords.append(pred)
    return list_preds_per_coords