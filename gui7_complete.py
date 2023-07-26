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

newVoiceRate = 100
text_speech = pyttsx3.init()
text_speech.setProperty('rate', newVoiceRate)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#cap = cv2.VideoCapture(0)

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

class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Recognition App")

        self.label = tk.Label(self.master, text="Face Recognition App", font=("Helvetica", 16))
        self.label.pack(pady=10)

        self.video_frame = tk.Frame(self.master)
        self.video_frame.pack()

        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        self.start_button = tk.Button(self.master, text="Start", command=self.start)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(self.master, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(pady=10)

        self.train_button = tk.Button(self.master, text="Train", command=self.train)
        self.train_button.pack(pady=10)

        self.quit_button = tk.Button(self.master, text="Quit", command=self.quit)
        self.quit_button.pack(pady=10)

        self.video_capture = None
        self.process_this_frame = True

    def start(self):
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

        self.video_capture = cv2.VideoCapture("http://192.168.43.1:4747/video")
        self.video_capture = cv2.VideoCapture(0)

        self.process_this_frame = True

        def process_frame():
            ret, frame = self.video_capture.read()
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
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
                            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                            # aa = ['Name'].values
                            # tt = str(Id)+"-"+aa
                            attendance.loc[len(attendance)] = [name, date, timeStamp]
                            #attendance = attendance.drop_duplicates(subset=['name'], keep='first')
                            fileName = "Attendance\Attendance_" + ".csv"
                            attendance.to_csv(fileName, mode='a', header=None, index=False)

                            arduino.sendData([1])
                            sleep(4)
                            arduino.sendData([0])
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

            self.process_this_frame = not self.process_this_frame
        self.stop_button.config(state="normal")

    def stop(self):
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.video_capture.release()
        cv2.destroyAllWindows()


    def quit(self):
        self.master.destroy()

    def train(self):
        self.train_button.config(state="disabled")
        print("[INFO] quantifying faces...")
        imagePaths = paths.list_images("ImagesAttendance")
        # initialize the list of known encodings and known names
        knownEncodings = []
        knownNames = []
        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            #print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
            name = imagePath.split(os.path.sep)[-2]
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recogn.face_locations(rgb, model="detection_method")
            encodings = face_recogn.face_encodings(rgb, boxes)

            # loop over the encodings
            for encoding in encodings:
                knownEncodings.append(encoding)
            knownNames.append(name)
        # dump the facial encodings + names to disk
        #print("[INFO] serializing encodings...")
        data = {"encodings": knownEncodings, "names": knownNames}
        f = open("encodings.pickle", "wb")
        f.write(pickle.dumps(data))
        f.close()
        print("[INFO] Training complete")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()