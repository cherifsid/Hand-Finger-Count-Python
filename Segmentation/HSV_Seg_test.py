import cv2 as cv
import numpy as np

img_path = "images/hands/hand3.png"
img = cv.imread(img_path)

hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")
skinRegionHSV = cv.inRange(hsvim, lower, upper)
blurred = cv.blur(skinRegionHSV, (2,2))
ret,thresh1 = cv.threshold(blurred,0,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(blurred,0,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(blurred,0,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(blurred,0,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(blurred,0,255,cv.THRESH_TOZERO_INV)


titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2,thresh3,thresh4, thresh5]
threshs = ['original image','thresh1','thresh2','thresh3','thresh4','thresh5',]

from matplotlib import pyplot as plt

for i in range(len(images)):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()
filtred_imges = [thresh1,thresh2,thresh4]
threshs = ['thresh1','thresh2','thresh4']

for i in range(len(filtred_imges)):
  contours, hierarchy = cv.findContours(filtred_imges[i], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  contours = max(contours, key=lambda x: cv.contourArea(x))
  cv.drawContours(img, [contours], -1, (255,255,0), 2)
  plt.subplot(2, 3, i + 1), plt.imshow(img, 'gray')
  plt.title(threshs[i])
  plt.xticks([])
  plt.yticks([])
plt.show()

#cv.imshow("thresh", thresh)