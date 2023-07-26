import cv2
import numpy as np
import face_recogn
import os
from datetime import datetime
import pickle
import pandas as pd
from tempfile import NamedTemporaryFile
import time
import datetime
# from PIL import ImageGrab
import openai_liveness3

from cvzone.SerialModule import SerialObject
from time import sleep

import pyttsx3
newVoiceRate = 100
text_speech = pyttsx3.init()
text_speech.setProperty('rate', newVoiceRate)

print("[INFO] loading encodings + face detector...")
data = pickle.loads(open("encodings.pickle", "rb").read())

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

arduino = SerialObject("COM9")

#def findEncodings(images):
    #encodeList = []
    #for img in images:
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #encode = face_recognition.face_encodings(img)[0]
        #encodeList.append(encode)
    #return encodeList

#def markAttendance(name):
    #with open('Attendance.csv','r+') as f:
        #myDataList = f.readlines()
        #nameList = []
        #for line in myDataList:
            #entry = line.split(',')
            #nameList.append(entry[0])
        #if name not in nameList:
            #now = datetime.now()
            #dtString = now.strftime('%H:%M:%S')
            #f.writelines(f'\n{name},{dtString}')

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

def recognize():

    cap = cv2.VideoCapture(0)
    col_names = ['name','date','time']
    attendance = pd.DataFrame(columns=col_names)

    while True:
        success, img = cap.read()
        #img = captureScreen()
        imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recogn.face_locations(imgS)
        encodesCurFrame = face_recogn.face_encodings(imgS, facesCurFrame)

        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recogn.compare_faces(data["encodings"], encodeFace)
            faceDis = face_recogn.face_distance(data["encodings"], encodeFace)
            #print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                #print(name)
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                #aa = ['Name'].values
                #tt = str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [name, date, timeStamp]
                attendance = attendance.drop_duplicates(subset=['name'], keep='first')

                arduino.sendData([1])
                sleep(4)
                arduino.sendData([0])
                text_speech.say("Attendance successfull" + str(name))
                text_speech.runAndWait()
        cv2.imshow('Frame2',img)
        if (cv2.waitKey(5000)) :
            break
        #key = cv2.waitKey(1)
        #if key == 27:
            #break

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    #attendance.drop(header_row)
    fileName = "Attendance\Attendance_"+date+".csv"
    attendance.to_csv( fileName, mode= 'a', header = None, index=False)

    cv2.destroyAllWindows()


#recognize()
