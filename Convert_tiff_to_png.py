import os 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def change_image_format_batch(src_path, tar_path, fmt_in, fmt_out ):
  '''
  Bildkonvertierung für alle Bilder im Stammordner von Ausgangformat in Zielforamt
  Args:
    src_path : 'string',  Stammordner,aus dem wir Bilder einlesen 
    tar_path : 'string', Stammordner, wo wir Ergebnis ablegen
    fmt_in  : 'string',  originale Bildformat
    fmt_out : 'string',  erwartete Bildformat
  '''
  import glob,os
  from PIL import Image

  if os.path.exists(src_path)==False:
    raise FileNotFoundError( 'No such file or directory:'+ src_path)
  
  img_dict=dict()
  directorys=[ subpath for subpath in os.listdir(src_path) if   os.path.isdir( os.path.join(src_path,subpath) )   ]
  #################################################
  if len(directorys)==0:
    imgPaths=glob.glob(os.path.join(src_path,'*.'+ fmt_in))
    if os.path.exists(tar_path)==False:
      os.makedirs(tar_path)
    for imgPath in imgPaths:
      im=Image.open(imgPath)
      _,imgNamePPM=os.path.split(imgPath)
      imgName,PPM=os.path.splitext(imgNamePPM)
      im.save(os.path.join(tar_path,imgName+'.'+ fmt_out))
    return


  #################################################
  for subdir in directorys:
    img_dict[subdir]=glob.glob(os.path.join(src_path,subdir,'*.'+ fmt_in))
  
  if os.path.exists(tar_path)==False:
    os.makedirs(tar_path)
  # erstelle Ordner
  for subdir,imgPaths in img_dict.items():
    newLongdir=os.path.join(tar_path,subdir)
    if os.path.exists(newLongdir)==False:
      os.makedirs(newLongdir)
    for imgPath in imgPaths:
      im=Image.open(imgPath)
      _,imgNamePPM=os.path.split(imgPath)
      imgName,PPM=os.path.splitext(imgNamePPM)
      im.save(os.path.join(tar_path,subdir,imgName+'.'+ fmt_out))

# konvertiere Bild von PPM in JPEG
## definiere Datenpfad
TIFF_IMAGE_ORDNER=r'Microtubules_Tiff_Date\August\200818\200818_Xb_Reaction2_6uM\000_30'
PNG_IMAGE_ORDNER=r'Microtubules_Tiff_Date\August\200818\200818_Xb_Reaction2_6uM\000_30_png'




# Prüfe, ob der Pfad existiert / korrekt eingegeben wurde
#assert os.path.exists(TIFF_IMAGE_ORDNER), "Der Pfad existriert nicht."

change_image_format_batch(TIFF_IMAGE_ORDNER,PNG_IMAGE_ORDNER,'tif','png')
