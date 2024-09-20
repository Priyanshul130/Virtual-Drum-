import cv2

cam=cv2.VideoCapture(0)

while True:

    status,frame=cam.read()
    frame=cv2.flip(frame,1)
    frame=cv2.resize(frame,(1280,720))
    
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)