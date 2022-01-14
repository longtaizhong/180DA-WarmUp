"""
Reference: 
https://answers.opencv.org/question/229620/drawing-a-rectangle-around-the-red-color-region/
https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
https://livecodestream.dev/post/object-tracking-with-opencv/

I referenced above websites for the basics of opencv, hsv, masking, and how contours work. 
Then I modified the code to detect the red object and draw rectangle on the largest contours. 
"""

import numpy as np
from cv2 import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    
    ret, frame = cap.read()

    #Convert to hsv, RED
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([98, 50, 50])
    upper_red = np.array([139, 255, 255])
    
    
    #Convert to rgb, RED
    #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #lower_red = np.array([120,0,0])
    #upper_red = np.array([255,100,100])
    #frame_threshed = cv2.inRange(rgb, lower_red, upper_red)

    frame_threshed = cv2.inRange (hsv, lower_red, upper_red)
    
    imgray = frame_threshed
    ret,thresh = cv2.threshold(frame_threshed,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Display the resulting frame
    #cv2.imshow('frame',frame)
    
    
    
    areas = [cv2.contourArea(c) for c in contours]
    
    max_index = np.argmax(areas)
        
    cnt=contours[max_index]
    
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Frame",frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()