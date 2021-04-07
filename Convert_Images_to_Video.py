import cv2
import os

image_folder = 'Microtubules_Tiff_Date/August/200818/200818_Xb_Reaction2_6uM/000_30_png'
video_name = '200818_xb_reaction2_6um0003.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()