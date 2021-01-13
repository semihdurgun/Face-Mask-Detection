import tensorflow as tf
import cv2
import numpy as np
from imutils.video import VideoStream
import imutils

interpreter = tf.lite.Interpreter("model-mask.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details() 

input_shape = input_details[0]['shape']
size  =  input_shape[1:3]

def detectMask(img,mainImg,startX,startY,endX,endY) :
    img = np.array(img, dtype=np.float32)
    img = cv2.resize(img, (300, 300))
    img = img / 255.0
    input_data = np.expand_dims(img , axis=0)
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    mask = predictions[0][0]
    withoutmask = predictions[0][1]
    label = "Mask" if mask> withoutmask else "No-Mask" 
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    label = "{}: {:.2f}%".format(label, max(mask, withoutmask) * 100)
    cv2.putText(mainImg, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(mainImg, (startX, startY), (endX, endY), color, 2)
    
    
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = VideoStream(src=0).start() 

while True: 
    img = cap.read()
    img = imutils.resize(img, width=800)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1 ,4)
    
    for (x ,y ,w ,h) in faces:
        face = img[x:y, (x+w,y+h)]
        cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0),2)
        
    #if len(faces)   == 1:
    for(x,y,w,h) in faces:
            face = img[y:y+h, x:x+w]
            face = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            face = cv2.resize(face,(300,300))
            detectMask(face,img,x,y,x+w,y+h)
        
    cv2.imshow("Frame", img)    
    key = cv2.waitKey(30) & 0xff

     # if the `q` key was pressed, break from the loop
    if key == ord("z"):
        break


cv2.destroyAllWindows() 
cap.stop()    