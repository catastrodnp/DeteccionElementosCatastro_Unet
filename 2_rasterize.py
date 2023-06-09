# -*- coding: utf-8 -*-
"""
@author: SSALAZAR
"""

import rasterio as rio
from rasterio.plot import show
import numpy as np
from osgeo import gdal
import os
from osgeo import ogr

def shp_raster(shp,folder,pixel=1,burn=255):
    layer=os.path.basename(shp)
    layer=os.path.splitext(layer)[0]
    print(layer)
    #Rasters
    raster = '{}/{}_{}cms.tif'.format(folder,layer,int(pixel*100))
    source_ds = ogr.Open(shp)
    source_layer = source_ds.GetLayer()
    x_min, x_max, y_min, y_max = source_layer.GetExtent()#Get extent from the shapefile
    print(source_layer.GetExtent())
    cmd='gdal_rasterize -l {} -burn {} -tr {} {} -te {} {} {} {} -ot Byte -of GTiff {} {}'.format(layer,burn,pixel,pixel,x_min,y_min,x_max,y_max,shp,raster)
    os.system(cmd)
    print(raster)