import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
import face_recognition

class FaceRecognitionThread(QThread):
    face_recognition_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.face_image = None
        self.known_face_encodings = []
        self.known_face_names = []

    def set_known_faces(self, known_face_encodings, known_face_names):
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names

    def set_face_image(self, face_image):
        self.face_image = face_image

    def run(self):
        while True:
            if self.face_image is not None:
                rgb_frame = cv2.cvtColor(self.face_image, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = self.known_face_names[first_match_index]

                    face_names.append(name)

                self.face_recognition_signal.emit(', '.join(face_names))

            self.msleep(2000)  # Wait for 2 seconds

class FaceRecognitionWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.video = cv2.VideoCapture(0)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.face_recognition_thread = FaceRecognitionThread()
        self.face_recognition_thread.face_recognition_signal.connect(self.handle_face_recognition)
        self.face_recognition_thread.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.video.read()

        if ret:
            self.face_recognition_thread.set_face_image(frame)

            height, width, channel = frame.shape
            bytes_per_line = channel * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            self.label.setPixmap(pixmap)
            self.label.setScaledContents(True)

    def handle_face_recognition(self, names):
        print("Recognized face(s):", names)

    def closeEvent(self, event):
        self.video.release()
        event.accept()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Recognition App")
        self.setCentralWidget(FaceRecognitionWidget())

if __name__ == "__main__":
    # Load known faces and encodings
    known_image = face_recognition.load_image_file("saikat.jpg")
    known_face_encoding = face_recognition.face_encodings(known_image)[0]
    known_face_names = ["Saikat "]

    app = QApplication(sys.argv)
    window = MainWindow()
    window.centralWidget().face_recognition_thread.set_known_faces([known_face_encoding], known_face_names)
    window.show()
    sys.exit(app.exec_())
