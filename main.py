import cv2
import numpy as np
from playsound import playsound



k_top,k_bottom,k_left,k_right=340,440,540,640#kick drum

h_top,h_bottom,h_left,h_right=340,440,300,400#hit hat drum

s_top,s_bottom,s_left,s_right=340,440,750,850#snare drum

k1_top,k1_bottom,k1_left,k1_right=480,580,640,740#kick drum 1

h1_top,h1_bottom,h1_left,h1_right=480,580,400,500#hit hat drum 1

s1_top,s1_bottom,s1_left,s1_right=480,580,850,950#snare drum 1


#get refrence to camera
cam=cv2.VideoCapture(0)

snare_drum=cv2.imread("snare_drum.png")
snare_drum=cv2.resize(snare_drum,(s_bottom-s_top, s_right-s_left))

kick_drum=cv2.imread("kick_drum.jpg")
kick_drum=cv2.resize(kick_drum,(k_bottom-k_top, k_right-k_left))

hithat_drum=cv2.imread("hithat_drum.jpg")
hithat_drum=cv2.resize(hithat_drum,(h_bottom-h_top, h_right-h_left))

#color to detect drum stick
lower=np.array([100,60,60])
upper=np.array([140,255,255])

#define a 5*5 slider for erosion and dilation
slider = np.ones((5, 5), np.uint8)



#loop till user press q

while True:
    #read a frame from a camera
    status,frame=cam.read()
    frame=cv2.flip(frame,1)
    frame=cv2.resize(frame,(1280,720))
    
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #determine which pixel fall within the blue boundries
    blue_mask=cv2.inRange(hsv,lower,upper)
    blue_mask = cv2.erode(blue_mask, slider, iterations=2)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, slider)
    blue_mask = cv2.dilate(blue_mask, slider, iterations=1)

    #find contour in the image
    (cnts,_) = cv2.findContours(blue_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #print(cnts)

    center=None
    if cnts is not None:
        #check to see if any contor were found
        if len(cnts)>0:
            #will assume its revelant contor at index 0
            cnt=cnts[0]
            #get the radious of the enclosing circlr around the found contor
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            
            #draw the cicle around the contour
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 0), 2)
            M = cv2.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            # if tounching drums
            if center[0]>=k_left and center[0]<=k_right and center[1]>=k_top and center[1]<=k_bottom:
                playsound("kick.mp3")

            elif center[0]>=h_left and center[0]<=h_right and center[1]>=h_top and center[1]<=h_bottom:
                playsound("hihat.mp3")

            elif center[0]>=s_left and center[0]<=s_right and center[1]>=s_top and center[1]<=s_bottom:
                playsound("snare.mp3")


            elif center[0]>=k1_left and center[0]<=k1_right and center[1]>=k1_top and center[1]<=k1_bottom:
                playsound("kick.mp3")

            elif center[0]>=h1_left and center[0]<=h1_right and center[1]>=h1_top and center[1]<=h1_bottom:
                playsound("hihat.mp3")

            elif center[0]>=s1_left and center[0]<=s1_right and center[1]>=s1_top and center[1]<=s1_bottom:
                playsound("snare.mp3")
    
    #get the three drum region(live screen)
    reg_kick = frame[k_top:k_bottom, k_left:k_right]
    reg_hithat = frame[h_top:h_bottom, h_left:h_right]
    reg_snare = frame[s_top:s_bottom, s_left:s_right]

    reg_kick1 = frame[k1_top:k1_bottom, k1_left:k1_right]
    reg_hithat1 = frame[h1_top:h1_bottom, h1_left:h1_right]
    reg_snare1 = frame[s1_top:s1_bottom, s1_left:s1_right]


    #add drums on live screen
    frame[k_top:k_bottom,k_left:k_right]=cv2.addWeighted(reg_kick,0.60,kick_drum,0.40,0.0)

    frame[h_top:h_bottom,h_left:h_right]=cv2.addWeighted(reg_hithat,0.60,hithat_drum,0.40,0.0)

    frame[s_top:s_bottom,s_left:s_right]=cv2.addWeighted(reg_snare,0.60,snare_drum,0.40,0.0)

    
    frame[k1_top:k1_bottom,k1_left:k1_right]=cv2.addWeighted(reg_kick1,0.60,kick_drum,0.40,0.0)

    frame[h1_top:h1_bottom,h1_left:h1_right]=cv2.addWeighted(reg_hithat1,0.60,hithat_drum,0.40,0.0)

    frame[s1_top:s1_bottom,s1_left:s1_right]=cv2.addWeighted(reg_snare1,0.60,snare_drum,0.40,0.0)
    

    #display the frame
    cv2.namedWindow("virtual drum",cv2.WINDOW_AUTOSIZE)
    cv2.imshow("virtual drum",frame)

    #if user press (q) quit the programe
    if cv2.waitKey(1)==ord("q"):
        break


#release the camera
cam.release()
cv2.destroyAllWindows()
    
    
                    
    















