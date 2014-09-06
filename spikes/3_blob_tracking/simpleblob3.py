import numpy as np
import cv2


# http://pastebin.com/DvHErcj8
# http://dalywhyte.blogspot.co.uk/2013/06/simplecv-blob-detection.html
# http://derek.simkowiak.net/motion-tracking-with-python/

cap = cv2.VideoCapture('V2.mp4')
# cap = cv2.VideoCapture(0)

# fgbg = cv2.BackgroundSubtractorMOG2()
fgbg = cv2.BackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    imgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # http://opencvpython.blogspot.co.uk/2013/05/thresholding.html
    # ret,thresh = cv2.threshold(imgray,127,255,0)
    # ret,thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,3)
    # ret,thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret,thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY)

    # contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(frame, contours, -1, (255,255,255), 3)
    # hull = cv2.convexHull(contours)

    # cv2.drawContours(frame, contours, -1, (255,255,255), 3)
    # cv2.drawContours(frame, contours, -1, (255,255,255), 3) # 0 - draw only contour 0

    mask = np.zeros(fgmask.shape,np.uint8)
    if hierarchy is not None :
      for i in range(0, len(hierarchy[0])):
          h = hierarchy[0][i]
          if len(contours[i]) > 10:
            cv2.drawContours(mask, contours , i, (055,055,255), 1) # 0 - draw only contour 0
          # if h[3] == -1 :
          #   print h
          # Each hierarchy entry is Next, Previous, FirstChild, Parent
          # cv2.drawContours(mask,cnt,-1,255,3)

      #     x,y,w,h = cv2.boundingRect(cnt)
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
        # cv2.imshow('Features', im)

    # b = cv2.SimpleBlobDetector()

    #set parameter
    # b.setInt('blobColor',0) # or 255?
    # b.setDouble('maxArea',
    # b.setDouble('maxCircularity',
    # b.setDouble('maxConvexity',
    # b.setDouble('maxInertiaRatio',
    # b.setDouble('maxThreshold',
    # b.setDouble('minDistBetweenBlobs',
    # b.setDouble('minRepeatability',
    # b.setDouble('minThreshold',
    # b.setDouble('thresholdStep',

     # findContours, approxPoly

    cv2.imshow('mask2',mask)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
