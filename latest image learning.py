import os
from PIL import Image
import numpy as np
import cv2 as cv
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"")

face_cascade= cv.CascadeClassifier('C:\Python\OCV\FACE\data\haarcascade_frontalface_alt2.xml')
recognizer=cv.face.LBPHFaceRecognizer_create()

current_id =0
label_ids={}

y_labels=[]
x_train =[]

for root,dirs,files, in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            print (label,path)
            
            if not label in label_ids:
                label_ids[label]=current_id
                current_id +=1
    
            id_=label_ids[label]
            print(label_ids)
            
            #y_labels.append(label)
            #x_train.append(path)

            #training
            pil_image= Image.open(path).convert("L")
            size =(550,550)
            final_image=pil_image.resize(size,Image.Resampling.LANCZOS)
            image_array=np.array(final_image,"uint8")
            
            print(image_array)
            
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5,minNeighbors=5)
            
            for (x,y,w,h) in faces:
                roi=image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
                
print(y_labels)
print(x_train)

with open("faces_name.pickle","wb") as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("facesdatabase.yml")

