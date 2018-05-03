参考：

https://mmeysenburg.github.io/image-processing/07-edge-detection/

http://blog.csdn.net/chevroletss/article/details/49785403

http://opencv-python-tutroals.readthedocs.io/en/latest/index.html

#当图片名里有中文的时候，发现无法读取图片

#直观上感觉sobel对于简单的图形检测效果好，canny对于复杂的图形检测效果好，（如果只按上述链接的参数的话）

 #OSTU: http://blog.csdn.net/on2way/article/details/46812121   


 ```
 # -*- coding: utf-8 -*-
 import numpy as np
 import cv2,sys

 #SUGGESTION: k = 3, t = 210
 #k = eval(input("kernel size :"))
 #t = eval(input("threshold :"))
 filename = 'C:\\Users\\Administrator\\Desktop\\2.jpg'
 img = cv2.imread(filename)
 cv2.imshow("Original", img)
 k = 3
 img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 cv2.imshow("Grey",img_grey)

 blur = cv2.GaussianBlur(img_grey, (k, k), 0)
 (th, binary) = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)

 edgeX = cv2.Sobel(binary, cv2.CV_64F, 1, 0)
 edgeY = cv2.Sobel(binary, cv2.CV_64F, 0, 1)

 #convert the datatype
 edgeX = np.uint8(np.absolute(edgeX))
 edgeY = np.uint8(np.absolute(edgeY))
 edge = cv2.bitwise_or(edgeX, edgeY)

 print("OTSU_threshold",th)
 cv2.imshow("OTSU_sobel_edges",edge)
 cv2.waitKey(0)
 #adp
 def adp_sobel():
     global sobel_th
     (sobel_th, binary) = cv2.threshold(blur, sobel_th, 255, cv2.THRESH_BINARY_INV)
     edgeX = cv2.Sobel(binary, cv2.CV_64F, 1, 0)
     edgeY = cv2.Sobel(binary, cv2.CV_64F, 0, 1)
     #convert the datatype
     edgeX = np.uint8(np.absolute(edgeX))
     edgeY = np.uint8(np.absolute(edgeY))
     edge = cv2.bitwise_or(edgeX, edgeY)
     cv2.imshow("Adp_sobel_edges",edge)

 def adjust_th(v):
     global sobel_th
     sobel_th = v
     adp_sobel()

 cv2.namedWindow("Adp_sobel_edges", cv2.WINDOW_NORMAL)
 sobel_th = 30
 cv2.createTrackbar("th", "Adp_sobel_edges", sobel_th, 255, adjust_th)
 cv2.waitKey(0)


 lap = cv2.Laplacian(img_grey, cv2.CV_64F)
 lap = np.uint8(np.absolute(lap))
 cv2.imshow("Edge detection by Laplacaian", np.hstack([lap, img_grey])) #这行代码放juypter里就报错，spyder就不会

 def cannyEdge():
     global img_grey, minT, maxT
     edge = cv2.Canny(img_grey, minT, maxT)
     cv2.imshow("canny detection_edges", edge)

 def adjustMinT(v):
     global minT
     minT = v
     cannyEdge()

 def adjustMaxT(v):
     global maxT
     maxT = v
     cannyEdge()

 cv2.namedWindow("canny detection_edges", cv2.WINDOW_NORMAL)
 minT = 30
 maxT = 150

 cv2.createTrackbar("minT", "canny detection_edges", minT, 255, adjustMinT)
 cv2.createTrackbar("maxT", "canny detection_edges", maxT, 255, adjustMaxT)

 cannyEdge()
 print("here")
 cv2.waitKey(0)
 ```


 
