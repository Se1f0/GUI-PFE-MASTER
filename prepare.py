import os
import numpy as np
import warnings
import time
from tools import resample,load_itk_image,process_mask,lumTrans,load_dicom_scan,get_pixels_hu,binarize_per_slice,all_slice_analysis,fill_hole,two_lung_only

def step1_python(case_path):
    case = load_dicom_scan(case_path)
    case_pixels, spacing = get_pixels_hu(case)
    bw = binarize_per_slice(case_pixels, spacing)
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68,7.5])
        cut_num = cut_num + cut_step

    bw = fill_hole(bw)
    bw1, bw2, bw = two_lung_only(bw, spacing)
    return case_pixels, bw1, bw2, spacing

def savenpy(data_path,prep_folder):        
    resolution = np.array([1,1,1])
    name = data_path.split("/")[-1]
    outputFolder = prep_folder+"/"+name
    isExist=os.path.exists(outputFolder)
    if not isExist:
        os.makedirs(outputFolder)
    im, m1, m2, spacing = step1_python(data_path)
    Mask = m1+m2
    
    newshape = np.round(np.array(Mask.shape)*spacing/resolution)
    xx,yy,zz= np.where(Mask)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
    extendbox = extendbox.astype('int')



    convex_mask = m1
    dm1 = process_mask(m1)
    dm2 = process_mask(m2)
    dilatedMask = dm1+dm2
    Mask = m1+m2
    extramask = dilatedMask ^ Mask
    bone_thresh = 210
    pad_value = 170
    im[np.isnan(im)]=-2000
    sliceim = lumTrans(im)
    sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
    bones = sliceim*extramask>bone_thresh
    sliceim[bones] = pad_value
    sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
    sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                extendbox[1,0]:extendbox[1,1],
                extendbox[2,0]:extendbox[2,1]]
    sliceim = sliceim2[np.newaxis,...]
    np.save(os.path.join(outputFolder,name+'_clean.npy'),sliceim)
    print(name)
    return(sliceim)

def full_prep(path):
    warnings.filterwarnings("ignore")
    t0 = int(round(time.time() * 1000))
    prep_folder = "Saved"
    data_path = path
    print('starting preprocessing')
    scan = savenpy(data_path,prep_folder)
    t1= int(round(time.time() * 1000))
    t2=t1-t0
    print('end preprocessing dicom in S : ' + str(t2/1000) + ' M : ' + str(t2/1000/60))
    return scan

def savenpy_luna(id,luna_segment,luna_data,savepath):
    isClean = True
    resolution = np.array([1,1,1])
    name = id
    outputFolder = savepath+"/"+name
    isExist=os.path.exists(outputFolder)
    if not isExist:
        os.makedirs(outputFolder)
    Mask,origin,spacing,isflip = load_itk_image(os.path.join(luna_segment,name+'.mhd'))
    if isflip:
        Mask = Mask[:,::-1,::-1]
    newshape = np.round(np.array(Mask.shape)*spacing/resolution).astype('int')
    m1 = Mask==3
    m2 = Mask==4
    Mask = m1+m2
    
    xx,yy,zz= np.where(Mask)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
   
    if isClean:
        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2
        Mask = m1+m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        sliceim,origin,spacing,isflip = load_itk_image(luna_data)
        if isflip:
            sliceim = sliceim[:,::-1,::-1]
            print('flip!')
        sliceim = lumTrans(sliceim)
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = (sliceim*extramask)>bone_thresh
        sliceim[bones] = pad_value
        
        sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
        sliceim = sliceim2[np.newaxis,...]
        np.save(os.path.join(outputFolder,name+'_clean.npy'),sliceim)
        print(name)
        return(sliceim)

def preprocess_luna(path):
    t0 = int(round(time.time() * 1000))
    luna_segment = "dataset/seg-lungs-LUNA16"
    savepath = "Saved"
    luna_data = path
    print('starting preprocessing luna')
    id = path.split("/")[-1][0:-4]
    scan = savenpy_luna(id,luna_segment,luna_data,savepath)
    t1= int(round(time.time() * 1000))
    t2=t1-t0
    print('end preprocessing luna in S : ' + str(t2/1000) + ' M : ' + str(t2/1000/60))
    return scan
    
def preprocessScan(path):
    if ".mhd" in path:
        scan = preprocess_luna(path)
    else:
        scan = full_prep(path)
    return scan

