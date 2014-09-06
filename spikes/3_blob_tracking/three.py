import numpy as np
import cv2

# cap = cv2.VideoCapture('vtest.avi')
cap = cv2.VideoCapture(0)

fgbg = cv2.BackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)


    storage = cv.CreateMemStorage(0)
    contour = cv.FindContours(grey_image, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE)
    points = []

    # while contour:
    #  bound_rect = cv.BoundingRect(list(contour))
    #  contour = contour.h_next()

    #  pt1 = (bound_rect[0], bound_rect[1])
    #  pt2 = (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3])
    #  points.append(pt1)
    #  points.append(pt2)
    #  cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(255,0,0), 1

    

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
