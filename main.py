import cv2 as cv
import matplotlib.pyplot as plt

img_path = "images/hands/hand5.png"
img = cv.imread(img_path)

import numpy as np

hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lower = np.array([0, 40, 40], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")
skinRegionHSV = cv.inRange(hsvim, lower, upper)
blurred = cv.blur(skinRegionHSV, (4,4))
ret,thresh = cv.threshold(blurred,0,255,cv.THRESH_BINARY)
#plt.imshow(thresh)
#plt.show()

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = max(contours, key=lambda x: cv.contourArea(x))
cv.drawContours(img, [contours], -1, (120,255,0), 2)
#cv.imshow("contours", img)
#plt.imshow(img)
#plt.show()

hull = cv.convexHull(contours)
cv.drawContours(img, [hull], -1, (146, 43, 34), 2)
#plt.imshow(img)
#plt.show()
hull = cv.convexHull(contours, returnPoints=False)
defects = cv.convexityDefects(contours, hull)

if defects is not None:
  cnt = 0
for i in range(defects.shape[0]):  # calculate the angle
  s, e, f, d = defects[i][0]
  start = contours[s][0]
  end = contours[e][0]
  far = contours[f][0]
  a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
  b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
  c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
  angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  #      cosine theorem
  print("s = {}".format(s),"e = {}".format(e))
  if angle < np.pi / 2:  # angle less than 90 degree, treat as fingers
    cnt += 1
    cv.circle(img, far, 4, [0, 0, 255], -1)
    #cv.line(img,start,end,[20, 255, 70],1)
    #cv.line(img,far,d,[20, 255, 70],1)
if cnt > 0:
  cnt = cnt+1
cv.putText(img, str(cnt), (0, 50), cv.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255) , 2, cv.LINE_AA)

if(cnt>1):
   print("number of finguers founds =  {}".format(cnt))
else:
  print(" {} finguer founds ".format(cnt))
plt.imshow(img)
plt.savefig('result/5.png')
plt.show()
