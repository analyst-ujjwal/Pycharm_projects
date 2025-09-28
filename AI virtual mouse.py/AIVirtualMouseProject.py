import cv2
import numpy as np
import time
import HandTrackingModule as htm
import autopy
from numpy.random import default_rng

##########################
wCam, hCam = 640, 480
pTime = 0
##########################

cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)


while True:
    ret, img = cap.read()
    if ret != True:
        break
    cv2.imshow("Window", img)
    image = detector.findHands(img)

    lmList, bbox = detector.findPosition(img)



    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, "FPS: {:.2f}".format(fps), (10, 40),cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0),2)


    cv2.imshow("Window", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.destroyAllWindows()