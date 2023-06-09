# -*- coding: utf-8 -*-
"""
@author: SSALAZAR
"""
import cv2
import os
import tqdm
import numpy as np

def metrics(true, pred):
    
    list_true=sorted([true+i for i in os.listdir(true)])
    list_pred=sorted([pred+i for i in os.listdir(pred)])

    iou_list=[]
    dice_list=[]

    for t,p in zip(list_true,list_pred):
        
        y_true=cv2.imread(t)
        y_pred=cv2.imread(p)
        axes = (1,2) 
        intersection = np.sum(np.logical_and(y_pred, y_true), axis=axes) 
        union = np.sum(np.logical_or(y_pred, y_true), axis=axes)
        mask_sum = intersection + union
        smooth = 0.001
        iou = (intersection + smooth) / (union + smooth)
        dice=(2*iou)/(iou+1)
        iou_list.append(iou)
        dice_list.append(dice)
    
    return  pred, np.mean(iou_list), np.mean(dice_list)