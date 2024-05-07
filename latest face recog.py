#importing libraries to be use
import cv2 as cv
import pickle
import numpy as np


#defining what camera to use
cam = cv.VideoCapture(0)

#using the xml file for what kind of image detection to be use
face_cascade= cv.CascadeClassifier('C:\Python\OCV\FACE\data\haarcascade_frontalface_alt2.xml')
recognizer=cv.face.LBPHFaceRecognizer_create()

#reading the database to recognize
recognizer.read("facesdatabase.yml")

#creating empty labels to be use
labels={}

# opening the created name list 
with open("faces_name.pickle","rb") as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}
                
while (True):
    #reading values of camera
    check, frame=cam.read()

    
    #converting the camera reading to gray
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)


    #region of interest 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5,minNeighbors=5)
    for(x,y,w,h) in faces:
        #print (x,y,w,h)  
        roi_gray=gray[y:y+h,x:x+w]    #y-cord_start,ycord_end
        roi_color =frame[y:y+h,x:x+w]
        
        id_,conf= recognizer.predict(roi_gray)
        #finding match and printing names
        if conf>=50:
            print(id_,conf)
            print(labels[id_])
            font=cv.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(0,255,0)
            thick =2
            cv.putText(frame,name,(x,y),font,1,color,thick,cv.LINE_AA)
            #defining parameters for drawing a rectangle/square to the face    

        elif conf <=50:
            name="unknown person"
            font=cv.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(0,255,0)
            thick =2
            cv.putText(frame,name,(x,y),font,1,color,thick,cv.LINE_AA)

        color= (0,255,0) #defining color (B,G,R) = green
        thick = 2  #thickness of border
        width= x+w  #end_cord_x  location of the face + the width 
        height=y+h  #end_cord_y  location of the face + the height 
        cv.rectangle(frame,(x,y),(width,height),color,thick) #creating rectangle using the parameters

        #defining parameters for drawing a rectangle/square to the face    
        #color= (0,255,0) #defining color (B,G,R) = green
        #thick = 2  #thickness of border
        #width= x+w  #end_cord_x  location of the face + the width 
        #height=y+h  #end_cord_y  location of the face + the height 
        #cv.rectangle(frame,(x,y),(width,height),color,thick) #creating rectangle using the parameters
        
        
    #display the window for camera output
    cv.imshow('Camera', frame )


    #escape button to stop
    key = cv.waitKey(1)
    if key == 27:
        break

#closing windows
cam.release()
cv.destroyAllWindows()
