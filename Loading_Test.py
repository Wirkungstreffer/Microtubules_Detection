from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from PIL import Image
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt

#from nd2reader import ND2Reader
#import matplotlib.pyplot as plt

#with ND2Reader('200818_Xb_Reaction2_6uM002_seeds.nd2') as images:
#  plt.imshow(images[0])

image = cv2.imread("200818_xb_reaction2_6um002_seedsc1t1.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
normed = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#image = cv2.imread('200818_xb_reaction2_6um002_seedsc1t1.tif')
#image = np.asarray(image, dtype = np.float64)
cv2.imshow("Image", normed)
cv2.waitKey(0)

#from nd2reader import ND2Reader
#import matplotlib.pyplot as plt
#import numpy as np

#plt.rcParams["figure.figsize"] = [20,10]

#img = None

#with ND2Reader('200818_Xb_Reaction2_6uM003.nd2') as images:
#  img = images[0]

#plt.plot(img)

#with ND2Reader('200818_Xb_Reaction2_6uM002.nd2') as images:
  #width and height of the image
  #print('%d x %d px' % (images.metadata['width'], images.metadata['height']))

#with ND2Reader('200818_Xb_Reaction2_6uM002_seeds.nd2') as images:
  #plt.imshow(images[0])