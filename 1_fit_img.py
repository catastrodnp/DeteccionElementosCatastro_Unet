# -*- coding: utf-8 -*-
"""
@author: SSALAZAR
"""

import rasterio as rio
from rasterio.plot import show
import numpy as np
from osgeo import gdal
import os,glob
from osgeo import ogr

def fit_img(tif_old,shp,folder,pixel=0.5,target):
    layer=os.path.basename(tif_old)
    layer=os.path.splitext(layer)[0]#Layer solito sin el path ni extension
    print(layer)
    source_ds = ogr.Open(shp)
    source_layer = source_ds.GetLayer()
    x_min, x_max, y_min, y_max = source_layer.GetExtent()#Get extent from the shapefile
    print(source_layer.GetExtent())    
    #RGB new
    rgb_n = '{}/{}_{}_{}cms_cut.tif'.format(folder,layer,target,int(pixel*100))
    cmd='gdalwarp -tr {} {} -te {} {} {} {} {} {}'.format(pixel,pixel,x_min,y_min,x_max,y_max,tif_old,rgb_n)
    os.system(cmd)
