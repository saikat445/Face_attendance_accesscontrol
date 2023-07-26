import cv2
import face_recogn
import tkinter as tk
from tkinter import messagebox
import pickle
import os
import numpy as np

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

        self.quit_button = tk.Button(self.master, text="Quit", command=self.quit)
        self.quit_button.pack(pady=10)

        self.video_capture = None
        self.process_this_frame = True

    def start(self):
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

        self.video_capture = cv2.VideoCapture(0)

        self.process_this_frame = True

        while True:
            ret, frame = self.video_capture.read()

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            if self.process_this_frame:
                face_locations = face_recogn.face_locations(rgb_small_frame)
                face_encodings = face_recogn.face_encodings(rgb_small_frame, face_locations)

                for encodeFace, faceLoc in zip(face_encodings, face_locations):
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

            self.process_this_frame = not self.process_this_frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Display the video feed
            cv2.imshow('Video', frame)

        self.video_capture.release()
        cv2.destroyAllWindows()

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def stop(self):
        self.video_capture.release()
        cv2.destroyAllWindows()

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def quit(self):
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()