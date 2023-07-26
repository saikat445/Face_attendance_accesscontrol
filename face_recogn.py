import cv2
import face_recognition
import time
import tkinter as tk
from time import sleep
from tkinter import messagebox
import os
import pickle
from datetime import datetime
import pandas as pd
from cvzone.SerialModule import SerialObject
import pyttsx3
import datetime
import numpy as np
from imutils import paths

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
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

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        face_cropped = frame[y:y + h, x:x + w]

        # Calculate the ratio of the eye region to the face region
        eye_region_height = int(h * 0.2)
        eye_region_width = int(w * 0.4)
        eye_region_x = int(w * 0.3)
        eye_region_y = int(h * 0.4)
        eye_region = face_cropped[eye_region_y:eye_region_y + eye_region_height,
                     eye_region_x:eye_region_x + eye_region_width]

        # Calculate the average intensity of the pixels in the eye region
        eye_intensity = cv2.mean(eye_region)[0]

        # Check if the average intensity is above a threshold
        if eye_intensity > 100:
            cv2.putText(frame, "fake", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "real", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            col_names = ['name','date','time']
            attendance = pd.DataFrame(columns=col_names)
            imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(data["encodings"], encodeFace)
                faceDis = face_recognition.face_distance(data["encodings"], encodeFace)
                # print(faceDis)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    # print(name)
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M')
                    # aa = ['Name'].values
                    # tt = str(Id)+"-"+aa
                    attendance.loc[len(attendance)] = [name, date, timeStamp]
                    #attendance = attendance.drop_duplicates(subset=['name'], keep='first', inplace=True)
                    fileName = "Attendance\Attendance_" + ".csv"
                    attendance.to_csv(fileName, mode='a',header= None, index=False)
            attendance2 = pd.read_csv('Attendance\Attendance_.csv')
            df=pd.DataFrame(attendance2)
            df.drop_duplicates('time', keep='first',inplace=True)
            print(df)
            fileName = "Attendance\df" + ".csv"
            df.to_csv(fileName, mode='w', header=None, index=False)


    cv2.imshow("frame",frame)

    if cv2.waitKey(1)== ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





