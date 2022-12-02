import cv2 as cv

img_path = "images/hands/5.png"
img = cv.imread(img_path)

import numpy as np

hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lower = np.array([0, 40, 40], dtype="uint8")
upper = np.array([20, 255, 255], dtype="uint8")
skinRegionHSV = cv.inRange(hsvim, lower, upper)
blurred = cv.blur(skinRegionHSV, (4, 4))
ret, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY)

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = max(contours, key=lambda x: cv.contourArea(x))
cv.drawContours(img, [contours], -1, (120, 255, 0), 2)
hull = cv.convexHull(contours)
cv.drawContours(img, [hull], -1, (146, 43, 34), 2)
hull = cv.convexHull(contours, returnPoints=False)
defects = cv.convexityDefects(contours, hull)

if defects is not None:
    cnt = 0
for i in range(defects.shape[0]):  # calculate the angle
    print(defects.shape[0])
    s, e, f, d = defects[i][0]
    start = contours[s][0]
    end = contours[e][0]
    far = contours[f][0]

    a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
    c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
    print(angle)

    if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
        cnt += 1
        cv.line(img, start, far, [70, 70, 255], 4)
        cv.line(img, end, far, [70, 70, 255], 4)
        cv.circle(img, far, 4, [0, 0, 255], -1)
        cv.circle(img, end, 4, [70, 5, 70], -1)
        cv.circle(img, start, 4, [70, 5, 70], -1)

    elif angle == 2.1237579287889248:
        cnt = 1
        cv.line(img, start, far, [70, 70, 255], 4)
        cv.line(img, end, far, [70, 70, 255], 4)
        cv.circle(img, far, 4, [0, 0, 255], -1)
        cv.circle(img, end, 4, [70, 5, 70], -1)
        cv.circle(img, start, 4, [70, 5, 70], -1)

if cnt == 1 :
  cv.line(img, start, far, [70, 70, 255], 4)
  cv.line(img, end, far, [70, 70, 255], 4)
  cv.circle(img, far, 4, [0, 0, 255], -1)
  cv.putText(img, str(cnt), (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

elif cnt > 1 :
    cnt = cnt + 1
    cv.line(img, start, far, [70, 70, 255], 4)
    cv.line(img, end, far, [70, 70, 255], 4)
    cv.circle(img, far, 4, [0, 0, 255], -1)
    cv.putText(img, str(cnt), (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

elif cnt == 0:
    cv.line(img, start, far, [70, 70, 255], 4)
    cv.line(img, end, far, [70, 70, 255], 4)
    cv.circle(img, far, 4, [0, 0, 255], -1)
    cv.putText(img, str(cnt), (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

import matplotlib.pyplot as plt

plt.imshow(img)
plt.savefig('results/5.png')
plt.show()

