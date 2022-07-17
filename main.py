import tensorflow as tf
from keras import models
import cv2
import time
import numpy as np
import os

#parameter
width_size = 640
height_size = 480
trainFolder = "CanData"

#Video stream
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,width_size)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height_size)

#model load
model = models.load_model("MobileNet.h5")

#Dataset folder name
className = sorted(["Can","Empty"])

def main():

    #Video processing
    while True:
       previousTime = time.time()
       ret , frame = cap.read()
       count = 0
       if ret:
         count += 1
         #Resize img
         processImage = frame.copy()
         processImage = cv2.cvtColor(processImage, cv2.COLOR_BGR2RGB)
         processImage = cv2.resize(processImage, (224,224))
         processImage = processImage/255.0
         processImage = np.expand_dims(processImage, axis=0)
         
         #Make prediction
         prediction = model.predict(processImage,verbose ="0")
         index = np.argmax(prediction)
         prob = round(prediction[0][index]*100,1)
         
         #Time for processing
         duration = time.time() - previousTime
         fps = int(1/duration)
         
         #Draw rectangle
         cv2.rectangle(frame,(0,0), (140,75), (0,0,0),-1)
         #Put text
         cv2.putText(frame,str(f"FPS:{fps}"),(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
         cv2.putText(frame,str(f"Class: {className[index]}"),(20,45),cv2.FONT_HERSHEY_SIMPLEX,0.5,(66,152,245),1)
         cv2.putText(frame,str(f"Prob: {prob}%"),(20,65),cv2.FONT_HERSHEY_SIMPLEX,0.5,(23,252,72),1)
         #Show window
         cv2.imshow("Detection",frame)
         key = cv2.waitKey(20)
         if key == ord('q'):
            print("Recording end!")
       	    break
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()