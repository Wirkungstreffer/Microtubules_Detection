from nd2reader import ND2Reader
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = [20,10]

with ND2Reader('200818_Xb_Reaction2_6uM003.nd2') as images:
  img = images[0]


#with ND2Reader('200818_Xb_Reaction2_6uM002_seeds.nd2') as images:
    # width and height of the image
    #print('%d x %d px' % (images.metadata['width'], images.metadata['height']))

#with ND2Reader('200818_Xb_Reaction2_6uM002_seeds.nd2') as images:
#  plt.imshow(images[0])