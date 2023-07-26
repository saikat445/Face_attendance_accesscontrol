import cv2
import numpy as np
import face_recogn
import os
from datetime import datetime
import pickle
# from PIL import ImageGrab

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recogn.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)

#data = {"encodings": knownEncodings, "names": knownNames}
data = {"encodings": encodeListKnown}
f = open("encodings2.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
print('Encoding Complete')
