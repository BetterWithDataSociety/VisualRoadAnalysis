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
    ret,thresh = cv2.threshold(imgray,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(fgmask, contours, -1, (255,255,255), 3)

    for i in range(0, len(contours)):
        if (i % 2 == 0):
           cnt = contours[i]
           #mask = np.zeros(im2.shape,np.uint8)
           #cv2.drawContours(mask,[cnt],0,255,-1)
           x,y,w,h = cv2.boundingRect(cnt)
           cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
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

    cv2.imshow('frame',frame)
    cv2.imshow('mask',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
