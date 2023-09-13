import os,glob
import sys
from PIL import Image
from tqdm import tqdm_notebook as tqdmn
import cv2
import numpy as np


def big_img_join(PRED_BIG='./prediction/pred_big/', ORI_PATH="./data/ORI_PATH/", PRED_PATH="./prediction/predictions", h=100, w=100):
  OUTPUT_PATH=PRED_BIG
  big_list=glob.glob(ORI_PATH+'/*.tif')
  test_output=glob.glob(PRED_PATH+'/*.tif')

  for bi in tqdmn(big_list):
    print(bi)
    big_img=cv2.imread(bi, cv2.IMREAD_UNCHANGED)

    h_img = big_img.shape[1]
    w_img = big_img.shape[0]

    height = h
    width = w

    X_test_output = np.zeros( (w_img, h_img), dtype=np.uint8 )

    for small_img in test_output:
        small = cv2.imread(small_img, cv2.IMREAD_UNCHANGED)
        small= small[:, :, 0]
        ijk=small_img.split('_')[-6:-2]
        i, j, k = ijk[0],ijk[1],ijk[2]
        try:
            X_test_output[ int(i): int(i) + width, int(j): int(j) + height] = small
        except:
            pass
    X_img = Image.fromarray(X_test_output, mode='L')
    X_img.save(os.path.join(PRED_BIG, bi.split('/')[-1]))
