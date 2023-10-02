import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
import os

def shp_regularize(shp_in,shp_out):

  gdf = gpd.read_file(shp_in)

  # Regularization and polygonization to each geometry
  regularized_geometries = []

  for index, row in gdf.iterrows():
      # Regularize the geometry (optional)
      regularized_geometry = row['geometry'].simplify(0.5, preserve_topology=True)
      
      # Convert the geometry to a Polygon if it's not already
      if not isinstance(regularized_geometry, Polygon):
          if regularized_geometry.geom_type == 'MultiPolygon':
              # Handle MultiPolygons by combining them into a single Polygon
              regularized_geometry = unary_union(regularized_geometry)
          elif regularized_geometry.geom_type == 'GeometryCollection':
              # Handle GeometryCollections by taking the largest Polygon
              regularized_geometry = max(
                  regularized_geometry, key=lambda geom: geom.area
              )
      
      # Add the regularized geometry to the list
      regularized_geometries.append(regularized_geometry)

  # Create a new GeoDataFrame with the regularized geometries
  gdf_regularized = gpd.GeoDataFrame({'geometry': regularized_geometries}, crs=gdf.crs)
  gdf_regularized.to_file(shp_out)
