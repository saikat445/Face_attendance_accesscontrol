from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTableWidgetItem,QPushButton
from PyQt5.QtCore import QTimer
import cv2
from keras.models import load_model
import pickle
import tensorflow as tf
import imutils
from keras.preprocessing.image import img_to_array
#import recognize
import face_recogn

from datetime import datetime
import time
import os
#from easyocr import Reader
#from openalpr import Alpr
import threading

import numpy as np
import pandas as pd
import pyttsx3

class FrameGrabber(QtCore.QThread):
    def __init__(self, parent=None):
        super(FrameGrabber, self).__init__(parent)

        #threading.Timer(5.0, self.take_snapshot).start()


    signal = QtCore.pyqtSignal(QtGui.QImage)

    def run(self):
        global cv2_im
        #global imgGray

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                cv2_im = frame
                cv2_im =self.app_to_obj_image(cv2_im)


                image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_BGR888)
                self.signal.emit(image)

    def app_to_obj_image(self,cv2_im):

        protoPath = "face_detector/deploy.prototxt"
        modelPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        # load the liveness detector model and label encoder from disk
        print("[INFO] loading liveness detector...")
        model_filepath = './liveness.model'
        model = tf.keras.models.load_model(
            model_filepath,
            custom_objects=None,
            compile=False
        )
        le = pickle.loads(open("le.pickle", "rb").read())

        try:
            frame = imutils.resize(cv2_im, width=600)
            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(cv2_im, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
        except AttributeError:
            pass

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections
            if confidence > 0.50:
                # compute the (x, y)-coordinates of the bounding box for
                # the face and extract the face ROI
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the detected bounding box does fall outside the
                # dimensions of the frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                # extract the face ROI and then preproces it in the exact
                # same manner as our training data
                face = cv2_im[startY:endY, startX:endX]
                face = cv2.resize(face, (32, 32))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                # pass the face ROI through the trained liveness detector
                # model to determine if the face is "real" or "fake"
                preds = model.predict(face)[0]
                j = np.argmax(preds)
                label = le.classes_[j]

                #if label == "real":
                    #recognize.recognize()
                # main()

                # draw the label and bounding box on the frame
                label = "{}: {:.4f}".format(label, preds[j])
                cv2.putText(cv2_im, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(cv2_im, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)

                #if label == "real":
                #self.now = datetime.now().strftime('%Y-%m-%d-%H%M%S')
                #obj_filename = f'''{self.now}'''
                #print(obj_filename)
                #self.roi = face
                #obj_path = f'''./NumberPlate/{obj_filename}.jpg'''
                #cv2.imwrite(f'''{obj_path}''', self.roi)
                #self.run_easyocr(obj_path)
                    #self.run_recognize(cv2_im)
        return cv2_im

    #def check_rego(self, plate_text):
        # write API or post request to check Government website
       # print(f'''{plate_text} being checked''')
        #print(f'''{plate_text} being written to CSV''')
        #f = open('Attendance/plate.csv', 'a')
        #f.write(f'''{plate_text},o,o\n''')
        #f.close()

    #def take_snapshot(self):
        #print('take snapshot init')
        # self.append_objs_to_img(cv2_im, True)
        #self.app_to_obj_image(cv2_im,True)
        #thread = threading.Timer(2.0, self.take_snapshot)
        #thread.daemon = True
        #thread.start()

    def run_recognize(self,cv2_im):
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

        newVoiceRate = 100
        text_speech = pyttsx3.init()
        text_speech.setProperty('rate', newVoiceRate)

        imgS = cv2.resize(cv2_im,(0,0),None,0.25,0.25)
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
                cv2.rectangle(cv2_im,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(cv2_im,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(cv2_im,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                #aa = ['Name'].values
                #tt = str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [name, date, timeStamp]
                attendance = attendance.drop_duplicates(subset=['Name'], keep='first')
                ts = time.time()
                #date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                #timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                Hour, Minute, Second = timeStamp.split(":")
                # attendance.drop(header_row)
                fileName = "Attendance\Attendance_" + date + ".csv"
                attendance.to_csv(fileName, mode='a', header=None, index=False)
                #arduino.sendData([1])
                #sleep(4)
                #arduino.sendData([0])
                text_speech.say("Attendance successfull" + str(name))
                text_speech.runAndWait()



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
            Anal.resize(800, 600)
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

            self.closed = QtWidgets.QPushButton(self.centralwidget)
            self.closed.setGeometry(QtCore.QRect(100, 480, 93, 28))
            self.closed.setObjectName("closed")
            Anal.setCentralWidget(self.centralwidget)

            self.tableWidget.setHorizontalHeaderLabels(['Plate', 'Rego', 'Sanc'])
            self.tableWidget.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
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

    #def refresh_table(self):
        #df = pd.read_csv('Attendance/Attendance.csv')
        #df = df.iloc[-10:].sort_index(ascending=False)
        #for each_row in range(len(df)):
            #self.tableWidget.setItem(each_row, 0, QTableWidgetItem(df.iloc[each_row][0]))
            #self.tableWidget.setItem(each_row, 1, QTableWidgetItem(df.iloc[each_row][1]))
            #self.tableWidget.setItem(each_row, 2, QTableWidgetItem(df.iloc[each_row][2]))

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
    #timer.timeout.connect(ui.refresh_table)
    timer.start(5000)

    sys.exit(app.exec_())