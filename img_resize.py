import os
from os import listdir
from PIL import Image as Img
import numpy as np
import cv2

face_cascades = cv2.CascadeClassifier('FaceRecog\haarcascades\haarcascade_frontalface_default.xml')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "appliData")
save_target_dir = os.path.join(BASE_DIR, "resized_images")





for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                img = cv2.imread(path)
                faces = face_cascades.detectMultiScale(img,1.3,4)

                for (x1,y1,w,h) in faces:
                
                    face = img[y1:y1+w, x1:x1+h]
                    resized = cv2.resize(face, (250,250), interpolation= cv2.INTER_LINEAR)
                    print(f"{os.path.join(save_target_dir,file[:-4])}-edited.png")
                    cv2.imwrite(f"{os.path.join(save_target_dir,file[:-4])}-edited.png", resized)

    
