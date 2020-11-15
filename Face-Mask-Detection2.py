
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from keras.models import  load_model
model = model_from_json(open("mask_detection2.json", "r").read())
model.load_weights('mask_detection2.h5')
facecascade=cv2.CascadeClassifier(r'C:\Users\TARUN\.PyCharmCE2019.1\config\scratches\haarcascades\haarcascade_frontalface_default.xml')
source=cv2.VideoCapture(0)
address='https://192.168.43.1:8080/video'
source.open(address)
labels_dict={1:'MASK',0:'NO MASK'}
color_dict={1:(0,255,0),0:(0,0,255)}
while (True):

    ret, img = source.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y + w, x:x + w]
        resized = cv2.resize(face_img, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(img, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('LIVE', img)
    key = cv2.waitKey(1)

    if (key == 27):
        break

cv2.destroyAllWindows()
source.release()
