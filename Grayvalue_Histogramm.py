from matplotlib import pyplot as plt
import cv2


array_of_img = []
def read_directory(directory):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+ directory):
        #print(filename) #just for test
        #img is used to store the image data 
        tiff_image = cv2.imread(directory + "/" + filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        img = cv2.normalize(tiff_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        array_of_img.append(img)
        #cv2.imshow("Image", img)
        #cv2.waitKey(0)
        #print(img)


tiff_image = cv2.imread("200818_xb_reaction2_6um002_seedsc1t1.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
#image = cv2.normalize(tiff_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#imgsize = image.shape

image = cv2.imread("1.png")

# gray grade
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original",gray)

#图像直方图
hist = cv2.calcHist([gray],[0],None,[256],[0,256])

plt.figure()#新建一个图像
plt.title("Grayscale Histogram")#图像的标题
plt.xlabel("Bins")#X轴标签
plt.ylabel("# of Pixels")#Y轴标签
plt.plot(hist)#画图
plt.xlim([0,256])#设置x坐标轴范围
plt.show()#显示图像