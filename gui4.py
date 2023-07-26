from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtCore import QTimer
import cv2

from tempfile import NamedTemporaryFile
import time
import datetime
#from datetime import datetime
#from easyocr import Reader
#from openalpr import Alpr
import threading

import numpy as np
import pandas as pd
import face_recogn
import os
import pickle

#import pytesseract
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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

        threading.Timer(15.0, self.take_snapshot).start()

    signal = QtCore.pyqtSignal(QtGui.QImage)

    def run(self):
        global cv2_im
        global imgGray
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #plateCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
        #minArea = 1000
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                cv2_im = frame
                #cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
                #imgGray = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2GRAY)
                cv2_im2 = self.append_objs_to_img(cv2_im,False)

                image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_BGR888)
                self.signal.emit(image)

    def append_objs_to_img(self,cv2_im,frame):
        #img_float32 = np.float32(cv2_im)
        #cv2_im_rgb = cv2.cvtColor(cv2_im.astype(np.int16), cv2.COLOR_BGR2RGB)
        cv2_im = cv2.resize(cv2_im, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recogn.face_locations(imgS)
        encodesCurFrame = face_recogn.face_encodings(imgS, facesCurFrame)
        col_names = ['Name', 'date', 'time']
        attendance = pd.DataFrame(columns=col_names)

        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recogn.compare_faces(data["encodings"], encodeFace)
            faceDis = face_recogn.face_distance(data["encodings"], encodeFace)
            #print(faceDis)
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
                timeStamp = datetime.datetime.fromtimestamp(
                    ts).strftime('%H:%M:')
                # aa = ['Name'].values
                # tt = str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [name, date, timeStamp]
                #attendance.sort_values("First Name", inplace=True)
                attendance2 = attendance.drop_duplicates(subset=['Name'],keep = 'first')
                #attendance = attendance.drop_duplicates()
                fileName = "Attendance\Attendance.csv"
                attendance2.to_csv(fileName, mode='a', header=None, index=False)

                    # self.file_list.append(f'''./detected/{obj_filename}.jpg''')
        return cv2_im

    def take_snapshot(self):
        print('take snapshot init')
        #self.append_objs_to_img(cv2_im,True)
        self.append_objs_to_img(cv2_im,True)
        thread = threading.Timer(5.0, self.take_snapshot)
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

    def setupUi(self, Anal):
        Anal.setObjectName("Anal")
        Anal.resize(800, 480)
        self.centralwidget = QtWidgets.QWidget(Anal)
        self.centralwidget.setObjectName("centralwidget")
        self.Imglabel = QtWidgets.QLabel(self.centralwidget)
        self.Imglabel.setGeometry(QtCore.QRect(0, 0, 648, 480))
        self.Imglabel.setObjectName("Imglabel")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(640, 0, 200, 480))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setRowCount(15)

        self.tableWidget.setHorizontalHeaderLabels(['Plate', 'Date', 'Time'])
        self.tableWidget.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        Anal.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Anal)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        Anal.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Anal)
        self.statusbar.setObjectName("statusbar")
        Anal.setStatusBar(self.statusbar)

        self.retranslateUi(Anal)
        QtCore.QMetaObject.connectSlotsByName(Anal)

    def refresh_table(self):
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        fileName = "Attendance\Attendance.csv"
        df = pd.read_csv(fileName)
        df = df.iloc[-15:].sort_index(ascending=False)
        for each_row in range(len(df)):
            self.tableWidget.setItem(each_row, 0, QTableWidgetItem(df.iloc[each_row][0]))
            self.tableWidget.setItem(each_row, 1, QTableWidgetItem(df.iloc[each_row][1]))
            self.tableWidget.setItem(each_row, 2, QTableWidgetItem(df.iloc[each_row][2]))

    @QtCore.pyqtSlot(QtGui.QImage)
    def updateFrame(self, image):
        self.Imglabel.setPixmap(QtGui.QPixmap.fromImage(image))

    def retranslateUi(self, Anal):
        _translate = QtCore.QCoreApplication.translate
        Anal.setWindowTitle(_translate("Anal", "MainWindow"))
        #self.Imglabel.setText(_translate("Anal", "TextLabel"))

    def appExec(self):
        self.grabber.appExec()

from PyQt5.QtWidgets import QApplication, QMainWindow

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow(MainWindow)
    MainWindow.show()

    timer = QTimer()
    timer.timeout.connect(ui.refresh_table)
    timer.start(5000)

    sys.exit(app.exec_())