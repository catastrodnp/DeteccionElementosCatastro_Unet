import os
import re
import sys
from PIL import Image
from tqdm import tqdm_notebook as tqdmn
import cv2
import numpy as np


def big_img_join(PRED_BIG='./prediction/pred_big/', ORI_PATH="./data/ORI_PATH/", PRED_PATH="./prediction/predictions", h=100, w=100):
          
    OUTPUT_PATH=PRED_BIG
    big_list=os.listdir(ORI_PATH)
    reference_pattern = re.compile('^.*AnalyticMS_rgb_scaled_([0-9_]*)_.*.tif$')#válido para rgb
    #reference_pattern = re.compile('^.*AnalyticMS_([0-9_]*)_.*.tif$')
    
    for bi in tqdmn(big_list):
        big_img=cv2.imread(os.path.join(ORI_PATH, bi))
        
        h_img = big_img.shape[1]
        w_img = big_img.shape[0]
    
        height = h
        width = w
    
        X_test_output = np.zeros( (w_img, h_img), dtype=np.uint8 )
        test_output = [ i for i in os.listdir(PRED_PATH) if i.startswith(bi.split('.', 1)[0])]
        
        for small_img in test_output:
            small = Image.open(os.path.join(PRED_PATH, small_img))
            small = np.array(small)
            small= small[:, :, 0]
            i, j, k = re.findall(reference_pattern, small_img)[0].split('_')[0:3]
            try:
                X_test_output[ int(i): int(i) + width, int(j): int(j) + height] = small
            except:
                pass
        X_img = Image.fromarray(X_test_output, mode='L')
        X_img.save(os.path.join(PRED_BIG, bi))