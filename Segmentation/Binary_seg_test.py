import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('images/hands/hand2.png')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
threshs = ['original image','thresh1','thresh2','thresh3','thresh4','thresh5',]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()

for i in range(len(images)):
  contours, hierarchy = cv.findContours(images[i], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  contours = max(contours, key=lambda x: cv.contourArea(x))
  cv.drawContours(img, [contours], -1, (255,255,0), 2)
  plt.subplot(2, 3, i + 1), plt.imshow(img, 'gray')
  plt.title(threshs[i])
  plt.xticks([])
  plt.yticks([])
plt.show()

"""""""""
we can see that the binary segemtation methods don't give us a good result so far so we will updapte 
this method to get better result 
"""""""""
img = cv.GaussianBlur(img,(5,5),0)
ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(len(images)):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
