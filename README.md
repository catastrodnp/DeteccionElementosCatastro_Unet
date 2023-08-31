<a target="_blank" href="https://colab.research.google.com/drive/1A3KbD7-sAJbPmyh2mlURrKhMGxSh-RUK">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


# U-Net para Detección de elementos de catastro

Este repositorio contiene un ejemplo de implementación del algoritmo U-Net para la detección de vías, construcciones, cercas, muros, manzanas y remoción en masas en imágenes satelitales. 
El modelo U-Net es una de las arquitecturas de red neuronal convolucional más utilizadas en tareas de segmentación semántica.

## Contenido

- [Requisitos](#Requisitos)
- [Fuente](#Fuente-Conjunto-de-datos)
- [Ejemplos](#Ejemplos)
- [Diagrama de Arquitectura](#diagrama-de-arquitectura)
- [Estimación de ahorros](#Estimación-de-ahorros)
- [Reconocimientos](#Reconocimientos)


## Requisitos
•	Python 3.x

•	Bibliotecas de Python:

    - segmentation-models-pytorch==0.2.1
    
    - albumentations
    
    - ipywidgets
    
    - geopandas
    
    - leafmap
    
    - localtileserver

## Diagrama de Arquitectura
![U-Net](https://www.mdpi.com/remotesensing/remotesensing-09-00680/article_deploy/html/images/remotesensing-09-00680-g002.png)

## Fuente Conjunto de datos
•	Imágenes tipo Ultracam:

    Sensor: UAV cámara Soda
    
    Bandas: RGB 

    Resolución espacial: 10 cms (urbano), 50 cms (rural)
    
    Dtype: unit8
    
    Fuente: https://www.colombiaenmapas.gov.co/
    

## Ejemplo conjunto de datos de entrenamiento
![Conjunto de datos de entrenamiento](ejemplo_dataset.png)

## Estimación de ahorros
Uuna estimación de ahorros en tiempo y costos por el uso de este tipo de algoritmos:

- Urbano: Reducción tiempo (42%), Reducción costo (66%)
  
- Rural: Reducción tiempo (31%), Reducción costo (58%)

## Reconocimiento
El entrenamiento y la detección del modelo se hizo usando la arquitectura UNET:

  U-Net: Convolutional Networks for Biomedical Image Segmentation
  Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, 2015, Volume 9351
  ISBN : 978-3-319-24573-7
  Olaf Ronneberger, Philipp Fischer, Thomas Brox

