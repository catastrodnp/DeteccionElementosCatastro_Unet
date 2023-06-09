import os
import re
import sys
import subprocess as sub
from tqdm import tqdm_notebook as tqdmn


def georeference(ORI_PATH='./data/big_img/', PRED_BIG="./prediction/pred_big/", GEO_PATH="./prediction/georeference"):

    for img in tqdmn(os.listdir(ORI_PATH)):
        print(img)
        paths=[os.path.join(ORI_PATH,img),os.path.join(PRED_BIG,img)]
        salida_geo = [os.path.join(GEO_PATH, os.path.basename(paths[1]).replace('.tif', '_geo.tif'))]

        sub.call([sys.executable,
              'C:\\Users\\sebas\\Anaconda3\\envs\\detection\\Scripts\\gdal_calc.py',
              '-A', paths[0], 
              '-B', paths[1], 
              '--B_band=1',
              '--outfile={}'.format(salida_geo[0]),
              '--type=Byte',
              '--NoDataValue=0',
              '--calc=B*(A>0)'])