from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
#from WebCam import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow

from imutils.video import VideoStream
from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import tensorflow as tf
import sys
sys.setrecursionlimit(1500)


class FrameGrabber(QtCore.QThread):
    def __init__(self, parent=None):
        super(FrameGrabber, self).__init__(parent)
        #self.ap = argparse.ArgumentParser()
        #self.ap.add_argument("-m", "--model", type=str, required=True,
                        #help="path to trained model")
        #self.ap.add_argument("-l", "--le", type=str, required=True,
                        #help="path to label encoder")
        #self.ap.add_argument("-d", "--detector", type=str, required=True,
                        #help="path to OpenCV's deep learning face detector")
        #self.ap.add_argument("-c", "--confidence", type=float, default=0.5,
                        #help="minimum probability to filter weak detections")
        #self.args = vars(self.ap.parse_args())

        print("[INFO] loading face detector...")
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

    signal = QtCore.pyqtSignal(QtGui.QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                cv2_im = frame

                cv2_im = self.append_obj_image(cv2_im)

                image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_BGR888)
                self.signal.emit(image)

    def append_obj_image(self,cv2_im):
        print("[INFO] loading face detector...")
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
            frame = imutils.resize(cv2_im)
            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        except AttributeError:
            pass
        cv2_im = self.append_obj_image(cv2_im)
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
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
                face = frame[startY:endY, startX:endX]
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
                    #main()
                # draw the label and bounding box on the frame
                label = "{}: {:.4f}".format(label, preds[j])
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)

        return cv2_im

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
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.closed = QtWidgets.QPushButton(self.centralwidget)
        self.closed.setGeometry(QtCore.QRect(100, 480, 93, 28))
        self.closed.setObjectName("closed")
        Anal.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Anal)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        Anal.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Anal)
        self.statusbar.setObjectName("statusbar")
        Anal.setStatusBar(self.statusbar)
        self.closed.clicked.connect(self.quitApp)

        self.retranslateUi(Anal)
        QtCore.QMetaObject.connectSlotsByName(Anal)

    @QtCore.pyqtSlot(QtGui.QImage)
    def updateFrame(self, image):
        self.Imglabel.setPixmap(QtGui.QPixmap.fromImage(image))

    def retranslateUi(self,Anal):
        _translate = QtCore.QCoreApplication.translate
        Anal.setWindowTitle(_translate("Anal", "MainWindow"))
        self.Imglabel.setText(_translate("Anal", "TextLabel"))
        self.closed.setText(_translate("Anal", "Close App"))

    def quitApp(self):
        QtWidgets.QApplication.close()



if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())