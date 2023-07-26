import cv2
import time
import recognize
from time import sleep
import pandas as pd
from datetime import datetime
import pickle
from datetime import datetime
import numpy as np
import face_recogn
import os
#from cvzone.SerialModule import SerialObject
import datetime
import pyttsx3
newVoiceRate = 100
text_speech = pyttsx3.init()
text_speech.setProperty('rate', newVoiceRate)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

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

#arduino = SerialObject("COM9")

def main():
    def process_frame():
        ret, frame = cap.read()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
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
                col_names = ['name', 'date', 'time']
                attendance = pd.DataFrame(columns=col_names)
                imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

                facesCurFrame = face_recogn.face_locations(imgS)
                encodesCurFrame = face_recogn.face_encodings(imgS, facesCurFrame)

                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    matches = face_recogn.compare_faces(data["encodings"], encodeFace)
                    faceDis = face_recogn.face_distance(data["encodings"], encodeFace)
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
                        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                        # aa = ['Name'].values
                        # tt = str(Id)+"-"+aa
                        attendance.loc[len(attendance)] = [name, date, timeStamp]
                        attendance = attendance.drop_duplicates(subset=['name'], keep='first')
                        fileName = "Attendance\Attendance_" + ".csv"
                        attendance.to_csv(fileName, mode='a', header=None, index=False)

                       # arduino.sendData([1])
                        #sleep(4)
                        #arduino.sendData([0])
                        text_speech.say("Attendance successfull" + str(name))
                        text_speech.runAndWait()

        cv2.imshow('frame', frame)

    # Start processing frames
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= 0.5:
            process_frame()
            start_time = time.time()
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
