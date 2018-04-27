---
layout: post
tags: [python,ml,cv,他山之石]
---
https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/

`kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))`
见：http://blog.csdn.net/sunny2038/article/details/9137759

cv.WaitKey()函数的功能是程序暂停:

http://blog.csdn.net/gxiaob/article/details/9027799/



最大类间方差法ostu：

http://blog.csdn.net/abcsunl/article/details/60959914



```
# -*- coding: utf-8 -*-

from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
from matplotlib import pyplot as plt

# define the dictionary to identify
# each digit on the thermostat

DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

img = cv2.imread('/Users/k/Documents/example.jpg')
print(img.shape)

cv2.imshow("ori", img)

img = imutils.resize(img, height=500)
cv2.imshow("cur", img)

print(img.shape)

#cv2.waitKey(0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edge = cv2.Canny(blur, 50, 200, 255)

cv2.imshow("edge", edge)
#cv2.waitKey(0)
#上面的两个cv2.waitKey(0)要注释掉，才能运行这行下面的program
#最后有一个cv2.waitKey(0)就够了

cnts = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        displayCnt = approx
        break

warped = four_point_transform(gray, displayCnt.reshape(4, 2))
output = four_point_transform(img, displayCnt.reshape(4, 2))
'''
cv2.imshow("warped",warped)
cv2.imshow("output",output)
cv2.waitKey(0)
'''
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
digitCnts = []

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    if w >= 15 and (h >= 30 and h <= 40):
        digitCnts.append(c)

digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
digits = []

for c in digitCnts:
    (x, y, w, h) = cv2.boundingRect(c)
    roi = thresh[y:y + h, x:x + w]

    (roiH, roiW) = roi.shape
    (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
    dHC = int(roiH * 0.05)

    # 七个矩形，以top为例，左上角是（0，0），右下角是（w，dH）
    # 这个坐标系和正常的不太一样
    segments = [
        ((0, 0), (w, dH)),  # top
        ((0, 0), (dW, h // 2)),
        ((w - dW, 0), (w, h // 2)),
        ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),
        ((0,h//2),(dW,h)),
        ((w - dW, h // 2), (w, h)),
        ((0, h - dH), (w, h))
    ]

    on = [0] * len(segments)

    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        segROI = roi[yA:yB, xA:xB]
        total = cv2.countNonZero(segROI)
        area = (xB - xA) * (yB - yA)
        if total / float(area) > 0.5:
            on[i] = 1
    digit = DIGITS_LOOKUP[tuple(on)]
    digits.append(digit)
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.putText(output, str(digit), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

print(u"{}{}.{} \u00b0C".format(*digits))
cv2.imshow("Input", img)
cv2.imshow("Output", output)
cv2.waitKey(0)

```
