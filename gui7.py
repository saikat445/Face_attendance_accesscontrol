from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow
from datetime import datetime
import threading
#import pywhatkit
#import keyboard
import time
import datetime
from datetime import datetime
import face_recogn
import os
import pickle
import os
import numpy as np
import pyttsx3
import pandas as pd

newVoiceRate = 100
text_speech = pyttsx3.init()
text_speech.setProperty('rate', newVoiceRate)

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


class FrameGrabber(QtCore.QThread):
    def __init__(self, parent=None):
        super(FrameGrabber, self).__init__(parent)

        threading.Timer(10.0, self.process_snap).start()

    signal = QtCore.pyqtSignal(QtGui.QImage)

    def run(self):
        global cv2_im
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        while cap.isOpened():
            success, frame = cap.read()
            if success :
                cv2_im = frame
                cv2_im = self.append_obj_to_image(cv2_im,False)

                image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_BGR888)
                self.signal.emit(image)

    def append_obj_to_image(self,cv2_im,take_photo):
        #ret, frame = self.cv2_im
        gray_image = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(cv2_im, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_cropped = cv2_im[y:y + h, x:x + w]

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
                cv2.putText(cv2_im, "fake", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
            else:
                cv2.putText(cv2_im, "real", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                col_names = ['name', 'date', 'time']
                attendance = pd.DataFrame(columns=col_names)
                imgS = cv2.resize(cv2_im, (0, 0), None, 0.25, 0.25)
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
                        cv2.rectangle(cv2_im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(cv2_im, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        cv2.putText(cv2_im, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                        ts = time.time()
                        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                        # aa = ['Name'].values
                        # tt = str(Id)+"-"+aa
                        attendance.loc[len(attendance)] = [name, date, timeStamp]
                        # attendance = attendance.drop_duplicates(subset=['name'], keep='first')
                        fileName = "Attendance\Attendance_" + ".csv"
                        attendance.to_csv(fileName, mode='a', header=None, index=False)

        return cv2_im

    def process_snap(self):
        print('take snapshot init')
        self.append_obj_to_image(cv2_im,True)
        thread = threading.Timer(5.0, self.process_snap)
        thread.daemon = True
        thread.start()


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, MainWindow):
        super().__init__()
        self.MainWindow = MainWindow
        self.setupUi(self.MainWindow)
        self.grabber = FrameGrabber()
        self.grabber.signal.connect(self.updateFrame)
        self.grabber.start()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(640, 480)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.imgLabel = QtWidgets.QLabel(self.centralwidget)
        self.imgLabel.setGeometry(QtCore.QRect(0, 0, 640, 480))
        self.imgLabel.setObjectName("ImgLabel")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(640,0,200,480))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setRowCount(12)

        self.tableWidget.setHorizontalHeaderLabels(["Name",'time','date'])
        self.tableWidget.horizontalHeader().setSectionResizeMode(0,QtWidgets.QHeaderView.Stretch)
        self.tableWidget.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)



        #self.quitPushButton.clicked.connect(self.quitApp)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    @QtCore.pyqtSlot(QtGui.QImage)
    def updateFrame(self, image):
        self.imgLabel.setPixmap(QtGui.QPixmap.fromImage(image))

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ANAL"))
        self.imgLabel.setText(_translate("MainWindow", "TextLabel"))


    def appExec(self):
        self.grabber.appExec()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())