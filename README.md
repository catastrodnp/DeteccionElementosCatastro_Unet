
<a target="_blank" href="https://colab.research.google.com/github/catastrodnp/DeteccionElementosCatastro_Unet/blob/main/notebook/Catastro_DNP_UNet.ipynb">
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

Agradecimiento especial a la Dirección de infraestructura y energía sostenible que desarrollaron una primera versión del código Unet para detectar vías terciarias. El inventario de vías terciarias completo para todos los departamentos de Colombia se puede consultar en el siguiente link: 
  
  https://onl.dnp.gov.co/Paginas/IA-Vias-Terciarias.aspx

## Uso en QGIS con Deepness
En la carpeta onnx de este repositorio se encuentran archivos de modelos pre-entrenados en formato ONNX que pueden ser utilizados para realizar la segmentación en QGIS mediante el complemento Deepness. Esto permite realizar el análisis de elementos como vías y construcciones directamente desde QGIS, sin necesidad de utilizar o ejecutar scripts.

Pasos para utilizar los modelos en QGIS:


  •	Abrir QGIS.
  Ir a Complementos -> Administrar e Instalar Complementos.
  Buscar Deepness e instalarlo.
  Descargar los modelos ONNX:
  
 •	Navegar a la carpeta onnx en este repositorio.
  Descargar los archivos de modelo correspondientes a los elementos que desea segmentar (por ejemplo, vías, construcciones).
  Cargar los modelos en Deepness:
  
  •	En QGIS, abrir el panel de Deepness.
  Importar el modelo ONNX descargado.
  Aplicar el modelo a las imágenes:

•	Seleccionar la imagen satelital o raster sobre la que desea realizar la segmentación.
•	Configurar los parámetros necesarios en Deepness.
•	Ejecutar el proceso para generar la segmentación.

De esta manera, puede realizar análisis avanzados de elementos catastrales directamente en QGIS, integrando eficientemente los resultados en sus flujos de trabajo sin necesidad de conocimientos avanzados en programación o ejecución de scripts.
